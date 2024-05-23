import os
import numpy as np
import torch
import torch.cuda as tc
import torchvision
import argparse
import sys, traceback

# my stuff
sys.path.append("./")
import utility.utils as uu
from datetime import datetime, timedelta

# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# SimCLR
from simclr import SimCLR
from simclr.modules import NT_Xent, get_resnet
#from simclr.modules.vit import get_vit
from simclr.modules.transformations import TransformsSimCLR
from simclr.modules.sync_batchnorm import convert_model

from model import load_optimizer, save_encoder, save_model
from utils import yaml_config_hook

def train(
    args: object, 
    train_loader: torch.utils.data.DataLoader, 
    model: torch.nn.Module, 
    criterion: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler, 
    stepwise_scheduler: torch.optim.lr_scheduler._LRScheduler, 
    scaler: torch.cuda.amp.grad_scaler, 
    epoch: int):
    
    loss_epoch = 0
    ltl = len(train_loader.dataset)//train_loader.batch_size
    
    # Main Loop
    for step, (x, idxs, targets) in enumerate(train_loader):
        
        # Load final batch for caching, but do not make model calcs with it
        if step+1 == ltl:
            break
        
        optimizer.zero_grad()

        if args.tf_device == "gpu":
            tf = train_loader.dataset.apply_gpu_transforms
            
        #assert (x_i.is_cuda is True and x_j.is_cuda is True)
        with tc.amp.autocast(enabled = True):
            if args.loss_device == "gpu":
                losses = model(x, tf = tf, device = args.device)
                print(losses)
                loss = torch.mean(losses)
            elif args.loss_device == "cpu":
                z_i, z_j = model(x, tf = tf, device = args.device)
                loss = criterion(z_i, z_j)
                print(loss)
            else:
                ValueError("loss_device must be cpu or gpu.")
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 128)
            optimizer.step()

        if stepwise_scheduler is not None:
            clr = stepwise_scheduler.get_last_lr()[0]
            stepwise_scheduler.step()
        elif scheduler is not None:
            clr = scheduler.get_last_lr()[0]
        else:
            clr = str(args.lr)
            if str(args.optimizer) in ["Adam", "AdamW", "LARS"]:
                clr = str(args.lr)+" (base rate, managed by "+str(args.optimizer)+")"

        if args.log_losses is True and step % 10 == 0:
            info = {"epoch": epoch, "step": step, "loss": loss.item(), "datetime": datetime.now()}
            if args.reload is True:
                first = False
            else:
                first = (epoch == 0 and step == 0)
            uu.csv_logger(args.save_path+"/train.log", info, first = first, overwrite = first)

        if not args.low_verbosity and args.nr == 0 and step % max(1, ltl // 10) == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{ltl}], Loss: {loss.item()}, Current LR: {clr}")

        loss_epoch += loss.item()

    return loss_epoch/ltl

def main(gpu: torch.DeviceObjType, args: object):
    rank = args.nr * args.gpus + gpu

    if args.nodes > 1:
        dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
        tc.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == "STL10":
        train_dataset = torchvision.datasets.STL10(
            args.dataset_dir,
            split="unlabeled",
            download=True,
            transform=TransformsSimCLR(size=args.image_size, to_tensor=True),
        )
    elif args.dataset == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            args.dataset_dir,
            download=True,
            transform=TransformsSimCLR(size=args.image_size, to_tensor=True),
        )
    elif args.dataset == "Custom_2D":
        mp.set_sharing_strategy("file_system")
        cache_manager = mp.Manager()
        cache_manager.shared_cache = cache_manager.dict()

        train_dataset = uu.Custom_Dataset(
            base_dir = vars(args).get("dataset_base_dir", "./datasets/medical_2D/train"), 
            cpu_transforms = None,
            gpu_transforms = TransformsSimCLR(
                size = args.image_size,
                grayscale = args.grayscale,
                to_tensor = False),
            verbose = (not args.low_verbosity),
            shape_desired = (tuple(shape for shape in args.shape_desired) if "shape_desired" in vars(args) else None),
            data_format = args.data_format,
            test_nan_inf = True,
            normalizer = args.normalizer,
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = cache_manager
            )
    else:
        raise NotImplementedError

    if "random_subset" in vars(args) and args.random_subset is not None:
        size = float(args.random_subset)
        train_dataset.split_dataset("base", names=["train"], fracs=[size])
        train_dataset.set_used_subset("train")

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None
    
    # Dataloader
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset, 
        batch_size = args.batch_size,
        num_workers = (32 if args.opmode == "live_cache" else args.num_workers), 
        prefetch_factor = (1 if args.opmode == "live_cache" else args.prefetch_factor), 
        drop_last = (False if args.opmode == "live_cache" else True), 
        shuffle = True,
        pin_memory = (False if args.opmode == "live_cache" else True),
        persistent_workers = (False if args.opmode == "live_cache" else True))

    # initialize ResNet (or other network)
    if not "encoder_network" in vars(args):
        args.encoder_network = args.resnet
    if "resnet" in args.encoder_network:
        encoder = get_resnet(args.encoder_network, pretrained=False)
        n_features = encoder.fc.in_features  # get dimensions of fc layer
        #print(encoder.fc.in_features, encoder.fc.out_features)
    #elif "vit" in args.encoder_network: # FIXME
        #encoder = get_vit(args.encoder_network, pretrained=False)
        #n_features = encoder.head.out_features
        #n_features = encoder.heads.head.in_features  # get dimensions of heads (fc) layer
        #print(encoder.classifier, encoder.classifier[3])
        #n_features = encoder.classifier[3].out_features
        #n_features = encoder.classifier[2].out_features #####
    else:
        raise NotImplementedError

    # initialize model
    if args.dataparallel:
        args.gpus_dp = tc.device_count() # hacked together to properly use all gpus when only DataParallel is true
    model = SimCLR(encoder, args, n_features)
    if args.reload:
        model_fp = os.path.join(args.save_path, "checkpoint_{}.tar".format(args.epoch_num)) 
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        args.start_epoch = args.epoch_num
    #model = model.to(args.device)

    # optimizer / loss
    optimizer, scheduler, stepwise_scheduler = load_optimizer(args, model, len_of_dataset=len(train_dataset))
    if args.loss_device == "cpu":
        criterion = NT_Xent(int(args.batch_size/args.gpus_dp), args.temperature, args.gpus_dp) # Will only be used if args.loss_device is not "gpu"
    else:
        criterion = None

    # DDP / DP
    if args.dataparallel:
        #if "resnet" in args.encoder_network: # only resnets need bns exchanged - vits have only layernorms
        model = convert_model(model)
        model = DataParallel(model)
        print("DataParallel active")
    else:
        if args.nodes > 1:
            #if "resnet" in args.encoder_network: # only resnets need bns exchanged - vits have only layernorms
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DDP(model, device_ids=[gpu])

    model = model.to(args.device)
    #if "resnet" in args.encoder_network:
    scaler = tc.amp.GradScaler(enabled = True)
    #elif "vit" in args.encoder_network:
    #scaler = None

    #if args.nr == 0:
    #    writer = SummaryWriter()

    args.global_step = 0

    try:
        for epoch in range(args.start_epoch, args.epochs):
            args.current_epoch = epoch
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            lr = optimizer.param_groups[0]["lr"]
            
            ### Train loop
            start = datetime.now()
            epoch_loss = train(args, train_loader, model, criterion, optimizer, scheduler, stepwise_scheduler, scaler, epoch) #model, criterion
                #print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
            print("Epoch time:", ((datetime.now()-start) / timedelta(microseconds=1)) / 1e6, "s")
            ###

            if args.nr == 0 and scheduler:
                scheduler.step()

            if args.nr == 0 and epoch % args.enc_save_frequency == 0:
                save_encoder(args, model, epoch)

            if args.nr == 0 and hasattr(args, "model_save_frequency") and epoch % args.model_save_frequency == 0:
                save_model(args, model)

            if args.nr == 0:
                print(f"Epoch Summary [{epoch+1}/{args.epochs}]\t Avg Loss: {epoch_loss}")#\t lr: {round(lr, 5)}")

            if (epoch == 0 or (args.reload is True and epoch == args.epoch_num)) and args.opmode == "live_cache":
                train_loader = torch.utils.data.DataLoader(
                    dataset = train_dataset, 
                    batch_size = args.batch_size,
                    num_workers = 8, 
                    prefetch_factor = 1,
                    drop_last = True, 
                    shuffle = True,
                    pin_memory = True,
                    persistent_workers = True
                    )
                train_loader.dataset.set_caching_complete()
    except Exception as e:
        # If the training is interrupted or crashes
        save_model(args, model)
        print("Training interrupted, model was saved.")
        uu.csv_logger(args.save_path+"/crash.log", {"== Exception caught ==": str(traceback.format_exc())}, first = True, overwrite = True)
        raise

    ## end training
    save_encoder(args, model, "pretrained")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "Pretraining")
    parser.add_argument("-c", "--config", dest="cfg_file", default="./pkgs/SimCLR/config/default_config.yaml", help="Use as -c /path/to/config.yaml")
    cfg_choice = parser.parse_args()

    if cfg_choice.cfg_file == "./pkgs/SimCLR/config/default_config.yaml":
        print("Warning: No config supplied/Default config selected.")
    elif cfg_choice.cfg_file[-5:] != ".yaml":
        raise argparse.ArgumentError("Specified file does not have a YAML file extension and probably can not be parsed.")

    config = yaml_config_hook(cfg_choice.cfg_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    # Master address for distributed data parallel
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "8001"

    args.save_path = os.path.join("./logs_and_checkpoints/pretraining/", args.name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.num_gpus = tc.device_count()
    args.world_size = args.gpus * args.nodes

    print("Initialization is a bit slow - but never lose hope, friend.")

    if args.nodes > 1:
        print(f"Training with {args.nodes} nodes, waiting until all nodes join before starting training")
        print("Distributed training currently disabled.")
        #mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)
