import sys, os, traceback
sys.path.append("./")

import torch
import torchvision
import torchvision.transforms.functional as ttf
import torch.nn as nn
from torch.nn.parallel import DataParallel
import torch.cuda as tc

from utility import utils as uu
from finetuning_tasks import finetuning_task

import numpy as np
import random
import argparse
from copy import deepcopy
from tqdm.auto import tqdm
from typing import Callable

# Training and validation loop
def tv_loop(
    args: object, 
    train_loader: torch.utils.data.DataLoader, 
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.grad_scaler, 
    eval_metrics: Callable,
    epoch: int,
    verbose: bool = True
    ):
    
    epoch_loss = 0

    # Train
    dataset.set_state("train")
    dataset.set_used_subset(args.train_set)
    model.train()
    if epoch == 0:
        train_steps = int(len(train_loader.dataset)/args.batch_size)+1
    else:
        train_steps = int(len(train_loader.dataset)/args.batch_size)

    for step, (data, idx, targets) in enumerate(train_loader):

        # To device
        data = data.to(device = args.device, **forced_dtype)
        if isinstance(targets, list):
            targets = [target_mask.to(args.device) for target_mask in targets]
        else:
            targets = targets.to(args.device)

        # Reset gradient accumulation
        optimizer.zero_grad()

        # Do model calculation
        if args.device == "cpu":
            preds, losses = model(data, targets)
            loss = torch.mean(losses)
            steploss = loss.item()
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        else:
            with tc.amp.autocast(enabled=True):
                preds, losses = model(data, targets)
                loss = torch.mean(losses)
            steploss = loss.item()
            epoch_loss += loss.item()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Step-wise scheduler step
        if args.scheduler is not None and args.scheduling_interval == "step":
            scheduler.step()

        # Print, log
        if verbose and step % (1 if args.debug is True else max(1, train_steps // 10)) == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}], Step [{step+1}/{train_steps}], Loss: {steploss}")
            sys.stdout.flush()
        if args.log_losses is True:
            t_info = {"epoch": epoch+1, "step": step+1, "loss": steploss}
            first = (True if epoch == 0 and step == 0 else False)
            uu.csv_logger(args.save_path+"/train.log", t_info, first = first, overwrite = first)

        # Short training for faster debugging
        if "short_training" in vars(args) and args.short_training is True and step == 0:
            break

    # Validate
    dataset.set_state("val")
    dataset.set_used_subset(args.val_set)
    model.eval()
    pl, tl = [], []
    
    for step, (data, idx, targets) in tqdm(enumerate(val_loader), total=(len(dataset)//args.batch_size)+1):

        # To device
        data = data.to(device = args.device, **forced_dtype)
        if isinstance(targets, list):
            targets = [target_mask.to(args.device) for target_mask in targets]
        else:
            targets = targets.to(args.device)
        
        # Do model calculation
        if args.device == "cpu":
            preds, losses = model(data, targets)
        else:
            with tc.amp.autocast(enabled=True), torch.no_grad():
                preds, losses = model(data, targets)
        #loss = torch.mean(losses)
        pl.extend([preds])
        tl.extend([targets])

        # Short validation for faster debugging
        if "short_training" in vars(args) and args.short_training is True and step == 0:
            break

    # Global metrics calculation
    metrics = eval_metrics(pl, tl)
    avg_loss = epoch_loss/train_steps

    # Log
    if args.log_losses is True:
        v_info = {"epoch": epoch, "avg_epoch_train_loss": avg_loss}
        v_info.update({m: v for m, v in metrics.items() if not m.startswith("#")})
        first = (True if epoch == 0 else False)
        uu.csv_logger(args.save_path+"/val.log", v_info, first = first, overwrite = first)

    # Epoch-wise scheduler step
    if args.scheduler is not None and args.scheduling_interval == "epoch":
        scheduler.step()

    return avg_loss, v_info

def fold(
        args: object, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        test_loader: torch.utils.data.DataLoader,
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        scaler: torch.cuda.amp.grad_scaler,
        eval_metrics: Callable,
        verbose: bool = True
        ):

    # Do training and validation for a number of epochs
    for epoch in range(args.epochs):
        al, v_info = uu.tqdm_wrapper(
            it = tv_loop(
                args = args,
                train_loader = train_loader,
                val_loader = val_loader,
                model = model,
                optimizer = optimizer,
                scheduler = scheduler,
                scaler = scaler, 
                eval_metrics = eval_metrics,
                epoch = epoch,
                verbose = verbose
            ),
            w = verbose
        )

        # If we were live caching, set dataloaders to new, permanent settings
        if epoch == 0 and args.opmode == "live_cache":
            dataset.set_caching_complete(args.train_set)
            dataset.set_caching_complete(args.val_set)
            train_loader = torch.utils.data.DataLoader(
                dataset = dataset, 
                batch_size = args.batch_size,
                num_workers = args.num_workers, 
                prefetch_factor = args.prefetch_factor, 
                drop_last = True, 
                shuffle = True,
                pin_memory = True,
                persistent_workers = True)
            val_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size = args.batch_size,
                num_workers = args.num_workers, 
                prefetch_factor = args.prefetch_factor, 
                drop_last = False, 
                shuffle = True,
                pin_memory = True,
                persistent_workers = False)
            test_loader = val_loader
            
        epoch_report = '\n'.join([f'{k}: {v}' for k, v in v_info.items() if not k.startswith("#")])
        print(epoch_report)
        sys.stdout.flush()
    
    # Finally, test time
    dataset.set_state("test")
    dataset.set_used_subset(args.test_set)
    model.eval()
    pl, tl, = [], []

    for step, (data, idx, targets) in tqdm(enumerate(test_loader), total = (len(dataset)//args.batch_size)+1):

        # To device
        data = data.to(device = args.device, **forced_dtype)
        if isinstance(targets, list):
            targets = [target_mask.to(args.device) for target_mask in targets]
        else:
            targets = targets.to(args.device)
        
        # Do model calculation
        if args.device == "cpu":
            preds, losses = model(data, targets)
        else:
            with tc.amp.autocast(enabled=True), torch.no_grad():
                preds, losses = model(data, targets)
        #loss = torch.mean(losses)
        pl.extend([preds])
        tl.extend([targets])

    # Global metrics calculation
    metrics = eval_metrics(pl, tl)
    relevant_metrics = {k: v for k, v in metrics.items() if not k.startswith("#")}

    if args.log_losses is True:
        uu.csv_logger(args.save_path+"/test.log", relevant_metrics, first = True, overwrite = True)

    # Return last average training loss, average test loss and test performance
    return al, relevant_metrics

if __name__ == "__main__":

    try:
        # Get args via config
        parser = argparse.ArgumentParser(description = "Fine-tuning")
        parser.add_argument("-c", "--config", dest = "cfg_file", default = "./pkgs/FTE/config/default_config.yaml", help="Use as -c /path/to/config.yaml")
        ap = parser.parse_args()

        if ap.cfg_file == "./pkgs/FTE/config/default_config.yaml":
            print("Warning: No config supplied/Default config selected.")
        elif ap.cfg_file[-5:] != ".yaml":
            raise argparse.ArgumentError("Specified file does not have a YAML file extension and probably can not be parsed.")

        config = uu.yaml_config_hook(ap.cfg_file)
        for k, v in config.items():
            parser.add_argument(f"--{k}", default = v, type = type(v))

        args = parser.parse_args()

        # Defaults
        args.save_path = os.path.join("./logs_and_checkpoints/finetuning/", args.name)
        if not "debug" in vars(args):
            args.debug = False
        if not "test_nan_inf" in vars(args):
            args.test_nan_inf = True
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)

        if isinstance(vars(args).get("w_l", None), list):
            args.w_l = torch.Tensor(args.w_l)
        else:
            args.w_l = None
        if isinstance(vars(args).get("w_m", None), list):
            args.w_m = torch.Tensor(args.w_m)
        else:
            args.w_m = None

        # Seeding
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

        # Get task specific network and dataset
        print("Starting (this may take a moment) ...")
        if args.opmode == "live_cache":
            manager = torch.multiprocessing.Manager()
            manager.shared_cache = manager.dict()
            torch.multiprocessing.set_sharing_strategy("file_system")
        else:
            manager = None
        dataset, model, optimizer, scheduler, eval_metrics = finetuning_task(args, manager)
        scaler = tc.amp.GradScaler(enabled = True)

        # Enforce float32 from dataset if device is "cpu"
        forced_dtype = {"dtype": torch.float32} if args.device == "cpu" else {}

        # DEBUG
        if args.debug is True:
            os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
            torch.autograd.set_detect_anomaly(True)

        # Dataloaders
        dl_kwargs = {
            "prefetch_factor": (1 if args.opmode == "live_cache" else args.prefetch_factor), 
            "persistent_workers" : (False if args.opmode == "live_cache" else True)
        }
        if args.num_workers > os.cpu_count() - 1:
            print(f"Asked for {args.num_workers} workers, but only {os.cpu_count() - 1} are available (at least 1 should always remain free). The number of workers has been capped.")
        if args.debug is True:
            nw_te0, nw_ve0 = 0, 0
            filtered_kwargs = uu.filter_kwargs(filter = ["prefetch_factor", "persistent_workers"], **dl_kwargs)
        elif args.opmode == "live_cache":
            nw_te0, nw_ve0 = min(32, os.cpu_count() - 1), min(os.cpu_count() - 1, args.num_workers)
            filtered_kwargs = dl_kwargs
        else:
            nw_te0, nw_ve0 = args.num_workers, args.num_workers
            filtered_kwargs = dl_kwargs
        print(f"Number of workers:")
        print(f"Train E0/E1+ {nw_te0}/{args.num_workers}, Val/Test E0/E1+ {nw_ve0}/{args.num_workers}")
            
        train_loader = torch.utils.data.DataLoader(
            dataset = dataset, 
            batch_size = args.batch_size,
            num_workers = nw_te0, 
            drop_last = (False if args.opmode == "live_cache" else True), 
            shuffle = True,
            pin_memory = True,
            **filtered_kwargs)
        val_test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size = args.batch_size,
            num_workers = nw_ve0, 
            drop_last = False, 
            shuffle = True,
            pin_memory = True,
            **filtered_kwargs)

        # Do n folds, using the base_f{n}_* subsets, or do a single "fold" with the train, val and test set
        if vars(args).get("folds", None) is None:
            args.train_set, args.val_set, args.test_set = "train", "val", "test"
            last_avg_train_loss, relevant_metrics = fold(
                args = args, 
                train_loader = train_loader, 
                val_loader = val_test_loader,
                test_loader = val_test_loader,
                model = model, 
                optimizer = optimizer, 
                scheduler = scheduler,
                scaler = scaler,
                eval_metrics = eval_metrics,
                verbose = args.verbose
                )
        else:
            raise NotImplementedError

        print(relevant_metrics)

        # Save model
        out = os.path.join(args.save_path, "finetuned_model.tar")
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), out)
            print("Finetuned model saved.")
        else:
            torch.save(model.state_dict(), out)
            print("Finetuned model saved.")
    except KeyboardInterrupt:
        print("Interrupting, please wait (please actually do) ...")
        sys.stdout.flush()
        sys.exit(2)
    except Exception:
        tb = str(traceback.format_exc())
        uu.csv_logger(args.save_path+"/crash.log", content = {"emsg": tb}, first = True, overwrite=True)
        sys.stdout.write(tb)
        sys.stdout.flush()
        sys.exit(1)
    sys.exit(0)