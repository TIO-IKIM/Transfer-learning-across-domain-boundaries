import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import List
from utility import utils as uu

# Adapted from:
# https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py

class BinaryDiceLoss(nn.Module):
    """
    Binary Dice Loss. Targets must be one-hot encoded.
    """
    def __init__(self, reduction: str = 'mean'):
        super(BinaryDiceLoss, self).__init__()
        self.reduction = reduction

    def forward(self, prediction: torch.Tensor, target: torch.Tensor):

        intersection = torch.sum(prediction * target, dim = (1, 2))
        cardinality = torch.sum(prediction, dim = (1, 2)) + torch.sum(target, dim = (1, 2))

        loss = 1 - 2 * (intersection / cardinality)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            raise NotImplementedError

class DiceLoss(nn.Module):
    """
    Dice Loss. Targets must be one-hot encoded.
    """
    def __init__(self, classes: int, weights: torch.Tensor = None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.classes = classes
        if weights is None:
            self.weights = torch.Tensor([1. if c == 0 else 3. for c in range(self.classes)])
        else:
            self.weights = weights

    def forward(self, predictions: torch.Tensor, oh_targets: torch.Tensor):
        BinaryDice = BinaryDiceLoss(**self.kwargs)
        predictions = nnf.softmax(predictions, dim=1)
        nc = oh_targets.size()[1]

        total_loss = 0
        for i in range(nc):
            dice_loss = BinaryDice(predictions[:, i, :, :], oh_targets[:, i, :, :])
            if self.weights is not None:
                dice_loss *= self.weights[i]
            total_loss += dice_loss

        return total_loss / nc

class SegmentationLoss(torch.nn.Module):

    """
    Return a loss module for the segmentation tasks. Supports XE+DICE, but DICE is disabled by default (and not even calculated), because it has a universal tendency to degrade the learning process.
    This class guarantees that there are no overlapping target masks, and constructs a background at runtime If you already had one, this constructed background is going to be empty).

    'w_l' should be None (equal weighting) or a tensor containing class weights.
    'classes' should be the number of classes for which you have targets masks.
    If 'loss_for_background' is True, class 0 will be the computed background, and a loss will be calculated for it. If False, class 0 will instead be whatever class the first target mask is for.
    """

    def __init__(
        self, 
        classes: int, 
        l_xe: float = 1, 
        l_dice: float = 0, 
        w_l: torch.Tensor = None,
        loss_for_background: bool = True,
        force_semantic = True):

        super(SegmentationLoss, self).__init__()
        self.classes = classes
        self.loss_for_background = loss_for_background
        if w_l is None: # if no weights given, equal weights by default
            if loss_for_background is True:
                w_l = torch.Tensor([1 for c in range(self.classes+1)])
            else:
                w_l = torch.Tensor([1 for c in range(self.classes)])
        self.XE = nn.CrossEntropyLoss(weight = w_l, reduction = "mean", ignore_index = -1)
        self.DICE = DiceLoss(classes = classes, weights = w_l)
        self.l_xe = l_xe
        self.l_dice = l_dice
        self.force_semantic = force_semantic

    def forward(self, predictions: torch.Tensor, targets: List[torch.Tensor,]):

        # De facto number of classes (including background computed at runtime)
        nc = (self.classes + 1 if self.loss_for_background is True else self.classes)

        # Sanity check
        if nc == predictions.size()[1]:
            pass
        else:
            raise ValueError(f"The amount of de facto used classes ({nc}) must be equal to the number of provided predictions ({predictions.size()[1]}). If loss_for_background ({self.loss_for_background}) is True, the number of provided predictions should be one greater.")

        # Add a background target mask based on the other targets
        all_targets = [torch.ones_like(targets[0]).to(targets[0].device)]
        all_targets.extend(targets)

        #oht = [-1]
        #oht.extend([i for i, t in enumerate(targets) if any(t.flatten())])
        #print(sorted(list(set(oht))))
        
        if self.force_semantic is True:
            # Convert to onehot, last class index always has priority if two masks match in one location
            c_targets = torch.squeeze(self.classes - torch.argmax(torch.stack(tensors = all_targets[::-1], dim = 1), dim = 1), dim = 1)
        else:
            raise NotImplementedError

        # If we don't compute loss for the background, toss the computed background out by mapping it to -1
        # and mapping everything else to the indices [0, classes]
        if self.loss_for_background is False:
            c_targets -= 1
            c_targets.clamp(0, None) # DEBUG

        #print(sorted(list(set(c_targets.detach().cpu().numpy().flatten()))))
        #print(c_targets.size(), predictions.size())

        # XE
        xe_loss = self.XE.forward(
            torch.moveaxis(predictions.squeeze(dim = (1, 2, 3)), 1, -1).flatten(end_dim = -2), 
            c_targets.flatten()
            )

        # DICE
        if self.l_dice != 0:
            if self.loss_for_background is False:
                c_targets = c_targets.clamp(0, None) # DEBUG
            oh_targets = nnf.one_hot(c_targets, num_classes = nc).moveaxis(-1, 1)
            dice_loss = self.DICE.forward(predictions, oh_targets)
            return (self.l_xe * xe_loss + self.l_dice * dice_loss)/(self.l_xe + self.l_dice)
        else:
            return xe_loss

class MultiLabelCE(torch.nn.Module):

    """
    First, converts the target class from a long to a one-hot representation, allowing multiple classes to be true.
    Second, applies BXE across all classes for the loss calculation.
    """

    def __init__(self, classes: int, weight: torch.Tensor = None):
        super(MultiLabelCE, self).__init__()
        self.classes = classes
        if weight is not None:
            self.weight = weight
        else:
            self.weight = torch.Tensor([1]*self.classes)
        self.XE = nn.BCEWithLogitsLoss(weight = self.weight, reduction = "mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        targets = uu.long_to_onehot(l = targets, c = self.classes)
        loss = self.XE(logits, targets)
        return loss