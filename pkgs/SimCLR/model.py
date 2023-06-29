import os
import torch
import torch.optim
import numpy as np

from simclr import SimCLR
from simclr.modules import LARS


def load_optimizer(args, model, len_of_dataset):

    scheduler = None
    stepwise_scheduler = None

    ### OPTIMIZER
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)  # originally lr=3e-4 # TODO: LARS
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "LARS":
        # optimized using LARS with linear learning rate scaling - updated to sqrt learning rate scaling
        # (i.e. LearningRate = 0.3 × BatchSize/256) and weight decay of 10−6. -> adapted to original paper values
        learning_rate = 0.075 * np.sqrt(args.batch_size)
        optimizer = LARS(
            model.parameters(),
            lr=learning_rate,
            weight_decay=args.weight_decay,
            exclude_from_weight_decay=["batch_normalization", "BatchNorm", "bn", "bias"]
            )

        # "decay the learning rate with the cosine decay schedule without restarts"
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.epochs, eta_min=0, last_epoch=-1
        )
    else:
        raise NotImplementedError

    ### STEPWISE SCHEDULER
    if "stepwise_scheduler" in vars(args):
        if args.stepwise_scheduler == "CosineAnnealing":
            # Every 10% of the epoch, one cosine cycle completes
            tau = vars(args).get("annealing_time", 0.2)
            stepwise_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer = optimizer,
                                                                                        T_0 = int(tau * len_of_dataset/args.batch_size),
                                                                                        T_mult = 1,
                                                                                        eta_min = 0,
                                                                                        last_epoch = -1,
                                                                                        verbose=False)
        elif args.stepwise_scheduler == "Exponential":
            # Every full epoch, the learning rate goes down by factor 0.5
            alpha = vars(args).get("decay_per_epoch", 0.5)
            gamma = np.exp(np.log(alpha)/(len_of_dataset/args.batch_size))
            stepwise_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer,
                                                                        gamma = gamma,
                                                                        last_epoch = -1,
                                                                        verbose = False)
        elif args.stepwise_scheduler == "DecayingCosine":
            # Do cosine cycles and decay simultaneously
            alpha = vars(args).get("decay_per_epoch", 0.5)
            gamma = np.exp(np.log(alpha)/(len_of_dataset/args.batch_size))
            tau = vars(args).get("annealing_time", 0.2)
            T_0 = int(tau * len_of_dataset/args.batch_size)
            eta_min = 1e-6
            def stepwise_decaying_cosine(step):
                return eta_min + 0.5*(1-eta_min) * (1 + np.cos((step/T_0) * np.pi))
            stepwise_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer,
                                                                    lr_lambda = stepwise_decaying_cosine,
                                                                    last_epoch = -1,
                                                                    verbose = False)
        elif args.stepwise_scheduler == "None":
            pass
        else:
            raise NotImplementedError

    if scheduler and stepwise_scheduler:
        print("Cannot have scheduler and stepwise_scheduler active at the same time.")
        raise NotImplementedError

    return optimizer, scheduler, stepwise_scheduler


def save_model(args, model):
    out = os.path.join(args.save_path, "checkpoint_{}.tar".format(args.current_epoch))

    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)

def save_encoder(args, model, epoch):
    out = os.path.join(args.save_path, "encoder_"+str(epoch)+".tar")
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.encoder.state_dict(), out)
        print("Encoder checkpoint saved.")
    else:
        torch.save(model.encoder.state_dict(), out)
        print("Encoder checkpoint saved.")