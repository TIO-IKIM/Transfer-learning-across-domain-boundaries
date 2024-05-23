# imports
import sys, os
sys.path.append("./")

import numpy as np
import random
import torch
import torch.nn as nn
from torch.nn import functional as nnf
import torchvision
import torchvision.transforms as ttf
import pkgs.SimCLR.simclr.modules.sync_batchnorm.batchnorm as sbn
from pkgs.SimCLR.simclr.modules import LARS
from network import ClassificationNetwork, SegmentationNetwork

from utility import utils as uu
from utility import augmentations as ua
from utility import losses as ul
from utility import eval_metrics as em
from typing import List, Callable

# Find the base path, so even if tasks are called from a different script, they load properly
# There is probably a nice way to do this using pathlib, but it works, so ...
ap = "/".join(os.path.abspath(__file__).split("/")[:-3])

def augs_from_task(args: object):
    augs = {
        "BraTS": ua.BraTS_augs,
        "BraTS-M": ua.BraTS_augs,
        "BraTS-S": ua.BraTS_augs,
        "LiTS": ua.LiTS_augs,
        "LiTS-M": ua.LiTS_augs,
        "LiTS-S": ua.LiTS_augs,
        "CTBR-XXS": ua.CTBR_augs,
        "CTBR-XS": ua.CTBR_augs,
        "CTBR-S": ua.CTBR_augs,
        "CTBR": ua.CTBR_augs,
        #"PASCAL-VOC": ua.PASCAL_VOC_augs,
        #"PASCAL-VOC-M": ua.PASCAL_VOC_augs,
        #"PASCAL-VOC-S": ua.PASCAL_VOC_augs,
        "ImageNet-10XS": ua.ImageNet_augs,
        "ImageNet-100XS": ua.ImageNet_augs,
        "ImageNet-1kXS": ua.ImageNet_augs,
        "ImageNet-10S": ua.ImageNet_augs,
        "ImageNet-100S": ua.ImageNet_augs,
        "ImageNet-1kS": ua.ImageNet_augs,
        "ImageNet-10": ua.ImageNet_augs,
        "ImageNet-100": ua.ImageNet_augs,
        "ImageNet-1k": ua.ImageNet_augs,
        "CX8": ua.CX8_augs,
        "CX8-M": ua.CX8_augs,
        "CX8-S": ua.CX8_augs,
    }
    if args.task not in list(augs.keys()):
        raise NotImplementedError
    else:
        cpu_augs, gpu_augs = augs[args.task](
            crop_size = tuple(vars(args).get("shape_desired", (256, 256))),
            noise_injection = vars(args).get("noise_injection", None)
            )
    if args.no_augs is True:
        return None, None
    else:
        return cpu_augs, gpu_augs

def model_from_task(args: object, tf: Callable):

    # Complete the model with head/decoder, depending on task
    if "LiTS" in args.task:
        loss_criterion = ul.SegmentationLoss(classes = 2, w_l = vars(args).get("w_l", None))
        model = SegmentationNetwork(
            enc_name = args.model_name,
            enc_weights = args.model_path,
            classes = 3,
            loss_criterion = loss_criterion,
            tf = tf,
            encoder_freeze = vars(args).get("frozen_encoder", False),
        )

    elif "BraTS" in args.task:
        loss_criterion = ul.SegmentationLoss(classes = 3, w_l = vars(args).get("w_l", None), loss_for_background = True)
        model = SegmentationNetwork(
            enc_name = args.model_name,
            enc_weights = args.model_path,
            classes = 4,
            loss_criterion = loss_criterion,
            tf = tf,
            encoder_freeze = vars(args).get("frozen_encoder", False),
        )

    elif "CTBR" in args.task:
        loss_criterion = nn.CrossEntropyLoss(weight = vars(args).get("w_l", None), reduction="mean")
        model = ClassificationNetwork(
            enc_name = args.model_name, 
            enc_weights = args.model_path, 
            classes = 6,
            loss_criterion = loss_criterion, 
            tf = tf,
            frozen_encoder = vars(args).get("frozen_encoder", False))

    elif "CX8" in args.task:
        loss_criterion = ul.MultiLabelCE(weight = vars(args).get("w_l", None), classes = 15)
        model = ClassificationNetwork(
            enc_name = args.model_name, 
            enc_weights = args.model_path, 
            classes = 15,
            loss_criterion = loss_criterion, 
            tf = tf,
            frozen_encoder = vars(args).get("frozen_encoder", False))

    elif "PASCAL-VOC" in args.task: # DEBUG classno and lfb
        loss_criterion = ul.SegmentationLoss(classes = 20, w_l = vars(args).get("w_l", None), loss_for_background = True)
        model = SegmentationNetwork(
            enc_name = args.model_name,
            enc_weights = args.model_path,
            classes = 21,
            loss_criterion = loss_criterion,
            tf = tf,
            encoder_freeze = vars(args).get("frozen_encoder", False),
        )

    elif "ImageNet" in args.task:
        size = "".join([char for char in args.task.split("-")[-1] if char.isdigit() or char=="k"])
        nc = {
            "1k": 1000,
            "100": 100,
            "10": 10
            }
        loss_criterion = nn.CrossEntropyLoss(weight = vars(args).get("w_l", None), reduction="mean")
        model = ClassificationNetwork(
            enc_name = args.model_name, 
            enc_weights = args.model_path, 
            classes = nc[size], 
            loss_criterion = loss_criterion, 
            tf = tf,
            frozen_encoder = vars(args).get("frozen_encoder", False))

    else:
        raise NotImplementedError

    # If SyncBN was used, keep using it
    if args.syncbn is True:
        model = sbn.convert_model(model)
    if args.device == "cuda":
        model = nn.DataParallel(model)
    model.to(args.device)

    return model

def dataset_from_task(args: object, manager: torch.multiprocessing.Manager):
    
    if args.task == "LiTS":
        if "shape_desired" not in vars(args):
            print("Warning: Segmentation downstream tasks generally need a fixed output shape to be set, which should be >= 224x224, otherwise targets and model output may have different shapes. Since you supply no desired shape here, the implicit assumption is that your data augmentations somehow guarantee the correct shape (e.g. by putting a Resize into the deterministic part of your augmentations). If you do not, you will probably see an error during the first loss forward pass.")

        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/Clean_LiTS", 
            tvt = {
                "train": f"{ap}/datasets/Clean_LiTS/train/volumes",
                "val": f"{ap}/datasets/Clean_LiTS/val/volumes",
                "test": f"{ap}/datasets/Clean_LiTS/test/volumes",
            },
            num_masks = 2,
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = args.data_format,
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset._get_target = uu.get_LiTS_target.__get__(dataset)

    elif "LiTS" in args.task and not args.task == "LiTS":
        size = args.task.split("-")[-1]

        if "shape_desired" not in vars(args):
            print("Warning: Segmentation downstream tasks generally need a fixed output shape to be set, which should be >= 224x224, otherwise targets and model output may have different shapes. Since you supply no desired shape here, the implicit assumption is that your data augmentations somehow guarantee the correct shape (e.g. by putting a Resize into the deterministic part of your augmentations). If you do not, you will probably see an error during the first loss forward pass.")

        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/Clean_LiTS", 
            tvt = {
                "train_full": f"{ap}/datasets/Clean_LiTS/train/volumes",
                "val": f"{ap}/datasets/Clean_LiTS/val/volumes",
                "test": f"{ap}/datasets/Clean_LiTS/test/volumes",
            },
            num_masks = 2,
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = args.data_format,
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset._get_target = uu.get_LiTS_target.__get__(dataset)
        if size == "M":
            dataset.split_dataset(target = "train_full", names = ["train"], fracs = [0.5])
        elif size == "S":
            dataset.split_dataset(target = "train_full", names = ["train"], fracs = [0.1])
        else:
            raise NotImplementedError(f"Unknown size {size}.")

    elif args.task == "BraTS":
        if "shape_desired" not in vars(args):
            print("Warning: Segmentation downstream tasks generally need a fixed output shape to be set, which should be >= 224x224, otherwise targets and model output may have different shapes. Since you supply no desired shape here, the implicit assumption is that your data augmentations somehow guarantee the correct shape (e.g. by putting a Resize into the deterministic part of your augmentations). If you do not, you will probably see an error during the first loss forward pass.")

        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/BraTS", 
            tvt_by_filename = {
                "train": uu.getFileList(f"{ap}/datasets/BraTS/train", "_t1.npy"),
                "val": uu.getFileList(f"{ap}/datasets/BraTS/val", "_t1.npy"),
                "test": uu.getFileList(f"{ap}/datasets/BraTS/test", "_t1.npy"),
            },
            num_masks = 3,
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = args.data_format,
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset._get_target = uu.get_BraTS_target.__get__(dataset)
        dataset._load = uu.load_4_channel_MRI_image.__get__(dataset)

    elif "BraTS" in args.task and not args.task == "BraTS":
        size = args.task.split("-")[-1]

        if "shape_desired" not in vars(args):
            print("Warning: Segmentation downstream tasks generally need a fixed output shape to be set, which should be >= 224x224, otherwise targets and model output may have different shapes. Since you supply no desired shape here, the implicit assumption is that your data augmentations somehow guarantee the correct shape (e.g. by putting a Resize into the deterministic part of your augmentations). If you do not, you will probably see an error during the first loss forward pass.")

        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/BraTS", 
            tvt_by_filename = {
                "train_full": uu.getFileList(f"{ap}/datasets/BraTS/train", "_t1.npy"),
                "val": uu.getFileList(f"{ap}/datasets/BraTS/val", "_t1.npy"),
                "test": uu.getFileList(f"{ap}/datasets/BraTS/test", "_t1.npy"),
            },
            num_masks = 3,
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = args.data_format,
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset._get_target = uu.get_BraTS_target.__get__(dataset)
        dataset._load = uu.load_4_channel_MRI_image.__get__(dataset)
        if size == "M":
            dataset.split_dataset(target = "train_full", names = ["train"], fracs = [0.5])
        elif size == "S":
            dataset.split_dataset(target = "train_full", names = ["train"], fracs = [0.1])
        else:
            raise NotImplementedError(f"Unknown size {size}.")

    elif args.task == "CTBR":
        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/medical_2D",
            tvt = {
                "train": f"{ap}/datasets/medical_2D/train_wo6",
                "val": f"{ap}/datasets/medical_2D/val_wo6",
                "test": f"{ap}/datasets/medical_2D/test_wo6",
            },
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = ".npy",
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset._get_target = uu.get_CTBR_target.__get__(dataset)
        dataset._get_target_name = uu.get_CTBR_target_name.__get__(dataset)

    elif "CTBR" in args.task and not args.task == "CTBR":
        size = args.task.split("-")[-1]

        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/medical_2D",
            tvt = {
                "train": f"{ap}/datasets/medical_2D/{size}/train_wo6",
                "val": f"{ap}/datasets/medical_2D/val_wo6",
                "test": f"{ap}/datasets/medical_2D/test_wo6",
            },
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = ".npy",
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset._get_target = uu.get_CTBR_target.__get__(dataset)
        dataset._get_target_name = uu.get_CTBR_target_name.__get__(dataset)

    elif args.task == "PASCAL-VOC":
        if "shape_desired" not in vars(args):
            print("Warning: Segmentation downstream tasks generally need a fixed output shape to be set, which should be >= 224x224, otherwise targets and model output may have different shapes. Since you supply no desired shape here, the implicit assumption is that your data augmentations somehow guarantee the correct shape (e.g. by putting a Resize into the deterministic part of your augmentations). If you do not, you will probably see an error during the first loss forward pass.")

        base_dir = f"{ap}/datasets/PASCAL_VOC_12/VOC2012"

        with open(f"{ap}/datasets/PASCAL_VOC_12/VOC2012/ImageSets/Segmentation/train.txt") as o:
            trainfiles = [base_dir+"/JPEGImages/"+filename.rstrip('\n')+".jpg" for filename in o.readlines()]
        with open(f"{ap}/datasets/PASCAL_VOC_12/VOC2012/ImageSets/Segmentation/val.txt") as o:
            valfiles = [base_dir+"/JPEGImages/"+filename.rstrip('\n')+".jpg" for filename in o.readlines()]

        dataset = uu.Custom_Dataset(
            base_dir = base_dir, 
            tvt_by_filename = {
                "train": trainfiles,
                "valtest": valfiles,
            },
            num_masks = 21,
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = args.data_format,
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset._get_target = uu.get_PVOC_target.__get__(dataset)
        dataset.split_dataset(target = "valtest", names = ["val", "test"], fracs = [0.5, 0.5])

    elif "PASCAL-VOC" in args.task and not args.task == "PASCAL-VOC":
        size = args.task.split("-")[-1]
        
        if "shape_desired" not in vars(args):
            print("Warning: Segmentation downstream tasks generally need a fixed output shape to be set, which should be >= 224x224, otherwise targets and model output may have different shapes. Since you supply no desired shape here, the implicit assumption is that your data augmentations somehow guarantee the correct shape (e.g. by putting a Resize into the deterministic part of your augmentations). If you do not, you will probably see an error during the first loss forward pass.")

        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/PASCAL_VOC_12/VOC2012", 
            tvt_by_filename = {
                "train_all": trainfiles,
                "valtest": valfiles,
            },
            num_masks = 21,
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = args.data_format,
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset._get_target = uu.get_PVOC_target.__get__(dataset)
        if size == "M":
            dataset.split_dataset(target = "train_full", names = ["train"], fracs = [0.5])
        elif size == "S":
            dataset.split_dataset(target = "train_all", names = ["train"], fracs = [0.1])
        else:
            raise NotImplementedError(f"Unknown size {size}.")
        dataset.split_dataset(target = "valtest", names = ["val", "test"], fracs = [0.5, 0.5])

    elif args.task == "ImageNet-1k":
        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/imagenet-1k/", 
            tvt = {
                "train": f"{ap}/datasets/imagenet-1k/train/",
                "val": f"{ap}/datasets/imagenet-1k/our_val/",
                "test": f"{ap}/datasets/imagenet-1k/our_test/"
                },
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = ".jpeg",
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset.make_ImageNet_targets = uu.make_ImageNet_targets.__get__(dataset)
        dataset.make_ImageNet_targets(size = "1k")
        dataset._get_target = uu.get_ImageNet_target.__get__(dataset)
        dataset._get_target_name = uu.get_ImageNet_target_name.__get__(dataset)
 
    elif "ImageNet" in args.task and not args.task == "ImageNet-1k":
        size = args.task.split("-")[-1]
        cnum = {
            "1kXS": "1k", 
            "1kS": "1k", 
            "1k": "1k",
            "100XS": "100",
            "100S": "100", 
            "100": "100",
            "10XS": "10",
            "10S": "10",
            "10": "10",
            }
        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/imagenet-{size}/", 
            tvt = {
                "train": f"{ap}/datasets/imagenet-{size}/train/",
                "val": f"{ap}/datasets/imagenet-{cnum[size]}/our_val/",
                "test": f"{ap}/datasets/imagenet-{cnum[size]}/our_test/"
                },
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = ".jpeg",
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset.make_ImageNet_targets = uu.make_ImageNet_targets.__get__(dataset)
        dataset.make_ImageNet_targets(size = size)
        dataset._get_target = uu.get_ImageNet_target.__get__(dataset)
        dataset._get_target_name = uu.get_ImageNet_target_name.__get__(dataset)
   
    elif "CX8" in args.task:
        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/CX8/", 
            tvt = {
                "train_val": f"{ap}/datasets/CX8/train_val/",
                "test": f"{ap}/datasets/CX8/test/"
                },
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = ".png",
            grayscale_only = True,
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        dataset.split_dataset(target = "train_val", names = ["train", "val"], fracs = [0.9, 0.1])
        dataset.make_ImageNet_targets = uu.make_CX8_targets.__get__(dataset)
        dataset.make_ImageNet_targets()
        dataset._get_target = uu.get_CX8_target.__get__(dataset)

    elif "CX8" in args.task and not args.task == "CX8":
        size = args.task.split("-")[-1]
        dataset = uu.Custom_Dataset(
            base_dir = f"{ap}/datasets/CX8/", 
            tvt = {
                "train_val": f"{ap}/datasets/CX8/train_val/",
                "test": f"{ap}/datasets/CX8/test/"
                },
            cpu_transforms = args.cpu_augs,
            gpu_transforms = args.gpu_augs,
            verbose = (True if "verbose" not in vars(args) else args.verbose),
            shape_desired = (None if "shape_desired" not in vars(args) else args.shape_desired),
            data_format = ".png",
            grayscale_only = True,
            test_nan_inf = args.test_nan_inf,
            normalizer = "naive",
            opmode = args.opmode,
            tf_device = args.tf_device,
            manager = manager,
            seed = args.seed,
            debug = args.debug)
        if size == "M":
            dataset.split_dataset(target = "train_val", names = ["train", "val"], fracs = [0.5, 0.1])
        elif size == "S":
            dataset.split_dataset(target = "train_val", names = ["train", "val"], fracs = [0.1, 0.1])
        else:
            raise NotImplementedError(f"Unknown size {size}.")
        dataset.make_ImageNet_targets = uu.make_CX8_targets.__get__(dataset)
        dataset.make_ImageNet_targets()
        dataset._get_target = uu.get_CX8_target.__get__(dataset)

    else:
        raise NotImplementedError
    
    # If n-fold cross validation, make folds
    if vars(args).get("folds", None) is not None:
        raise NotImplementedError

    # If pre-caching, now is the time
    if args.opmode == "pre_cache":
        dataset._cache_dataset_zeroW(name = "base")

    return dataset

def get_optimizer(args: object, model: torch.nn.Module):
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "LARS":
        #lr = 0.075 * np.sqrt(args.batch_size)
        optimizer = LARS(
            model.parameters(), 
            lr=args.lr, 
            #lr=lr,
            weight_decay=args.weight_decay, 
            exclude_from_weight_decay=["batch_normalization", "BatchNorm", "bn", "bias"])
    else:
        raise NotImplementedError

    return optimizer

def get_scheduler(args: object, optimizer: torch.optim.Optimizer):
    if args.scheduler is None:
        scheduler = None
    elif args.scheduler == "cosine":
        # Perform cosine annealing (with warm restarts)
        if args.scheduling_interval == "step":
            tau = vars(args).get("annealing_time", 0.2)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer = optimizer,
                T_0 = int(tau * args.len_of_trainset/args.batch_size),
                T_mult = 1,
                eta_min = 0,
                last_epoch = -1,
                verbose=False)
        elif args.scheduling_interval == "epoch":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer = optimizer,
                T_0 = vars(args).get("annealing_time", 10),
                T_mult = 1,
                eta_min = 0,
                last_epoch = -1,
                verbose=False)
    elif args.scheduler == "decaying_cosine":
        # Do cosine cycles and decay simultaneously
        if args.scheduling_interval == "step":
            alpha = vars(args).get("decay_per_epoch", 0.99)
            gamma = np.exp(np.log(alpha)/(args.len_of_trainset/args.batch_size))
            tau = vars(args).get("annealing_time", 0.2)
            T_0 = int(tau * args.len_of_trainset/args.batch_size)
            eta_min = 1e-6
            def stepwise_decaying_cosine(step):
                return (eta_min + 0.5*(1-eta_min) * (1 + np.cos((step/T_0) * np.pi))) * (gamma**step)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer = optimizer,
                lr_lambda = stepwise_decaying_cosine,
                last_epoch = -1,
                verbose = False)
        elif args.scheduling_interval == "epoch":
            gamma = vars(args).get("decay_per_epoch", 0.99)
            T_0 = vars(args).get("annealing_time", 10)
            eta_min = 1e-6
            def decaying_cosine(epoch):
                return (eta_min + 0.5*(1-eta_min) * (1 + np.cos((epoch/T_0) * np.pi))) * (gamma**epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer = optimizer,
                lr_lambda = decaying_cosine,
                last_epoch = -1,
                verbose = False)
    elif args.scheduler == "decay":
        # Every full epoch, the learning rate goes down to factor decay_per_epoch
        if args.scheduling_interval == "step":
            alpha = vars(args).get("decay_per_epoch", 0.99)
            gamma = np.exp(np.log(alpha)/(args.len_of_trainset/args.batch_size))
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer = optimizer,
                gamma = gamma,
                last_epoch = -1,
                verbose = False)
        elif args.scheduling_interval == "epoch":
            gamma = vars(args).get("decay_per_epoch", 0.99)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer = optimizer,
                gamma = gamma,
                last_epoch = -1,
                verbose = False)
    else:
        raise NotImplementedError
    return scheduler

def metrics_from_task(args: object):
    
    def eval_metrics(predictions, targets):

        results = {}
        
        # Select appropriate metrics for given task
        if any(x in args.task for x in ["ImageNet", "CTBR"]):
            metrics = em.Classification_Metrics(
                verbose = args.verbose, 
                weights = vars(args).get("w_m", None))
        elif any(x in args.task for x in ["CX8"]):
            if "CX8" in args.task:
                num_classes = 15
            else:
                raise NotImplementedError
            metrics = em.MultiClass_Classification_Metrics(
                classes = num_classes, 
                verbose = args.verbose, 
                weights = vars(args).get("w_m", None))
        elif any(x in args.task for x in ["PASCAL-VOC", "LiTS"]):
            if "PASCAL-VOC" in args.task:
                num_classes = 20 # DEBUG classno and mfb
                metrics_for_background = True # No computed background
                report_avg = [x+1 for x in range(num_classes - 1)] # 1-20, since we also don't want the regular background in our average
            elif "LiTS" in args.task:
                num_classes = 2
                metrics_for_background = True
                report_avg = []
            else:
                raise NotImplementedError
            metrics = em.Segmentation_Metrics(
                verbose = args.verbose, 
                classes = num_classes, 
                weights = vars(args).get("w_m", None), 
                metrics_for_background = metrics_for_background,
                report_avg = report_avg)
        elif "BraTS" in args.task:
            metrics = em.BraTS_Metrics(
                verbose = args.verbose, 
                classes = 3, 
                weights = vars(args).get("w_m", None), 
                metrics_for_background = True,
                report_avg = [])
        else:
            raise NotImplementedError

        results.update(metrics.forward(predictions = predictions, targets = targets))

        return results

    return eval_metrics

def finetuning_task(args: object, manager: torch.multiprocessing.Manager):
    valid_tasks = [
        "BraTS",
        "BraTS-M",
        "BraTS-S",
        "LiTS",
        "LiTS-M",
        "LiTS-S",
        "CTBR-XXS",
        "CTBR-XS",
        "CTBR-S",
        "CTBR",
        "PASCAL-VOC",
        "PASCAL-VOC-M",
        "PASCAL-VOC-S",
        "ImageNet-10XS",
        "ImageNet-100XS",
        "ImageNet-1kXS",
        "ImageNet-10S",
        "ImageNet-100S",
        "ImageNet-1kS",
        "ImageNet-10",
        "ImageNet-100",
        "ImageNet-1k",
        "CX8",
        "CX8-M",
        "CX8-S",
    ]

    if args.task not in valid_tasks:
        raise NotImplementedError

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.cpu_augs, args.gpu_augs = augs_from_task(args)
    dataset = dataset_from_task(args, manager)
    tf = (dataset.apply_gpu_transforms if args.tf_device == "gpu" else None) # If we do random transforms on the GPU, we outsource the function to the model
    args.len_of_trainset = dataset.__len__(subset = "train")
    model = model_from_task(args, tf)
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    eval_metrics = metrics_from_task(args)

    return dataset, model, optimizer, scheduler, eval_metrics