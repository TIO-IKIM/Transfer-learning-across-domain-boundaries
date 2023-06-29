import sys
sys.path.append("./")

import torch
from torch.nn import functional as nnf
from typing import List, Union
from utility import utils as uu, losses as ul
from collections import OrderedDict
import numpy as np
from sklearn import metrics

"""
Every metric gets a callable class, that must take a list of tensors as input.
The init method should expect anything you want to add in the form of kwargs,
which you would also have to add to the finetuning_tasks.py file."
"""

class Classification_Metrics():

    """
    Accuracy and average loss.
    """

    def __init__(self, verbose: bool = False, **kwargs):
        self.results = OrderedDict()
        self.verbose = verbose
        self.kwargs = kwargs

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):

        loss_fn = torch.nn.CrossEntropyLoss()
        val_losses = []
        hits = 0
        seen = []

        for p, t in zip(predictions, targets):
            pc = torch.argmax(p, dim=1).flatten()
            hits += uu.truesum((pc == t).tolist())
            val_losses.append(loss_fn(p, t).item())
            seen.append(len(t))

        if self.verbose is True:
            print(f"[{hits}/{sum(seen)}] Hits/Seen")

        self.results["accuracy"] = hits/sum(seen)
        self.results["avg_val_loss"] = sum([v*s for v, s in zip(val_losses, seen)])/sum(seen)
        
        return self.results

class MultiClass_Classification_Metrics():

    """
    Dice Score, IoU, Precision, and Recall per class, plus overall average loss.
    A class counts as predicted when a threshold of .5 is met for its probability in the logits.
    """

    def __init__(self, classes: int, verbose: bool = False, **kwargs):
        self.results = OrderedDict()
        self.classes = classes
        self.verbose = verbose
        self.kwargs = kwargs
        pass

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        device = targets[0].device
        self.loss_fn = ul.MultiLabelCE(classes = self.classes, weight = None).to(device = device)
        self.val_losses = []
        tp = {c: 0 for c in range(self.classes)}
        fp = {c: 0 for c in range(self.classes)}
        tn = {c: 0 for c in range(self.classes)}
        fn = {c: 0 for c in range(self.classes)}
        seen = 0

        self.threshold = 0.5
        eps = 1e-6

        for b, (p, t) in enumerate(zip(predictions, targets)):
            for c in range(self.classes):
                ps = torch.sigmoid(p)[:,c]
                ts = uu.long_to_onehot(l = t, c = self.classes)[:,c]
                pb = torch.ceil(nnf.threshold(ps, self.threshold, 0)).to(dtype = torch.long).to(dtype = torch.bool)
                tb = ts.to(dtype = torch.bool)
                tp[c] += uu.truesum((pb * tb).tolist())
                fp[c] += uu.truesum((pb * ~tb).tolist())
                tn[c] += uu.truesum((~pb * ~tb).tolist())
                fn[c] += uu.truesum((~pb * tb).tolist())
            seen += len(pb)
            self.val_losses.append(self.loss_fn(p, t).item())

        for c in range(self.classes):
            # Dice
            if tp[c] == 0 and 2 * tp[c] + fp[c] + fn[c] == 0:
                self.results[f"#dice_{c}"] = np.nan
            else:
                self.results[f"#dice_{c}"] = 2 * tp[c] / (2 * tp[c] + fp[c] + fn[c] + eps)

            # IoU
            if tp[c] == 0 and tp[c] + fp[c] + fn[c] == 0:
                self.results[f"#iou_{c}"] = np.nan
            else:
                self.results[f"#iou_{c}"] = tp[c] / (tp[c] + fp[c] + fn[c] + eps)
            
            # Precision
            if tp[c] == 0 and tp[c] + fp[c] == 0:
                self.results[f"#precision_{c}"] = np.nan
            else:
                self.results[f"#precision_{c}"] = tp[c] / (tp[c] + fp[c] + eps)

            # Recall / TPR
            if tp[c] == 0 and tp[c] + fn[c] == 0:
                self.results[f"#recall_{c}"] = np.nan
            else:
                self.results[f"#recall_{c}"] = tp[c] / (tp[c] + fn[c] + eps)

            # FPR
            if fp[c] == 0 and fp[c] + tn[c] == 0:
                self.results[f"#fpr_{c}"] = np.nan
            else:
                self.results[f"#fpr_{c}"] = fp[c] / (fp[c] + tn[c] + eps)

        # AUC
        for c in range(self.classes):
            try:
                psla = torch.sigmoid(torch.cat([item for item in predictions], dim = 0)).cpu().numpy()
                tsla = uu.long_to_onehot(l = torch.cat([item for item in targets], dim = 0), c = self.classes).cpu().numpy()
                aucs = metrics.roc_auc_score(tsla, psla, average = None) # Multi-Label case
                self.results.update({f"#auc_{c}": auc for c, auc in enumerate(aucs)}) 
            except:
                self.results[f"#auc_{c}"] = np.nan
                raise
        try:
            auc_c = sum([1 for c in range(self.classes) if not np.isnan(self.results[f"#auc_{c}"])])
            self.results["auc_avg_overall"] = sum([self.results[f"#auc_{c}"] for c in range(self.classes) if not np.isnan(self.results[f"#auc_{c}"])]) / auc_c
        except:
            self.results["auc_avg_overall"] = np.nan

        # All other unweighted overall averages (ignore classes for which we had no targets (unlikely) or no predictions (very possible early on))
        dice_c = sum([1 for c in range(self.classes) if not np.isnan(self.results[f"#dice_{c}"])])
        self.results["dice_avg_overall"] = sum([self.results[f"#dice_{c}"] for c in range(self.classes) if not np.isnan(self.results[f"#dice_{c}"])]) / dice_c
        iou_c = sum([1 for c in range(self.classes) if not np.isnan(self.results[f"#iou_{c}"])])
        self.results["iou_avg_overall"] = sum([self.results[f"#iou_{c}"] for c in range(self.classes) if not np.isnan(self.results[f"#iou_{c}"])]) / iou_c
        precision_c = sum([1 for c in range(self.classes) if not np.isnan(self.results[f"#precision_{c}"])])
        self.results["precision_avg_overall"] = sum([self.results[f"#precision_{c}"] for c in range(self.classes) if not np.isnan(self.results[f"#precision_{c}"])]) / precision_c
        recall_c = sum([1 for c in range(self.classes) if not np.isnan(self.results[f"#recall_{c}"])])
        self.results["recall_avg_overall"] = sum([self.results[f"#recall_{c}"] for c in range(self.classes) if not np.isnan(self.results[f"#recall_{c}"])]) / recall_c

        return self.results

class Segmentation_Metrics():

    """
    Computes:
    Dice Score, IoU, Precision, Recall.

    Targets must be a List of List of Tensors.
    (Outer list has batches, inner list has each target as a separate tensor.)

    'classes' should be the number of classes for which you have targets masks.
    """

    def __init__(self, classes, verbose: bool = False, **kwargs):
        self.results = OrderedDict()
        self.verbose = verbose
        self.kwargs = kwargs
        self.weights = self.kwargs.get("weights", None)
        self.classes = classes
        self.report_avg = self.kwargs.get("report_avg", [])
        self.metrics_for_background = self.kwargs.get("metrics_for_background", True)

    def forward(self, predictions: List[torch.Tensor, ], targets: List[torch.Tensor, ]):
        
        eps = 1e-6
        bsl = {}
        # De facto number of classes (including background computed at runtime)
        nc = (self.classes + 1 if self.metrics_for_background is True else self.classes)

        for b in range(len(targets)):
            
            # Convert predictions to binary one-hot format for proper scoring
            p_arg = nnf.one_hot(torch.argmax(predictions[b].to(torch.float32), dim = 1), num_classes = self.classes + 1).moveaxis(-1, 1)

            # Build the background label target on the fly
            all_targets = [torch.ones_like(targets[b][0]).to(targets[b][0].device)]
            all_targets.extend(targets[b])

            # Guarantee that there is no overlap (last class index always has priority if multiple masks are 1 for a pixel)
            c_targets = torch.squeeze(self.classes - torch.argmax(torch.stack(tensors = all_targets[::-1], dim = 1), dim = 1))

            # Convert to one-hot
            oh_targets = nnf.one_hot(c_targets, num_classes = self.classes + 1).moveaxis(-1, 1)
            oh_targets = oh_targets[:, self.classes - nc + 1:, :, :]
            #print(f"{sum(oh_targets)}/{256**2}")

            # Iterate over all classes, with or without computed background
            for c in range(nc): 
                    
                target = oh_targets[:, c, :, :].type(torch.bool).squeeze()
                prediction = p_arg[:, c, :, :].type(torch.bool)
                intersection = torch.sum(prediction * target)
                p_cardinality = torch.sum(prediction)
                t_cardinality = torch.sum(target)
                cardinality = p_cardinality + t_cardinality
                union = torch.sum((prediction + target))

                bs = target.size()[0]

                if self.weights is None:
                    weight = 1
                else:
                    weight = self.weights[c]

                # Dice Score
                if intersection.item() == 0 and cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    dice_score = np.nan
                else:
                    # Regular case
                    dice_score = (2. * intersection / (cardinality + eps)).item()

                self.results[f"#dice_{b}_{c}"] = dice_score
                bsl[f"#dice_{b}_{c}"] = (np.nan if np.isnan(dice_score) else bs * weight)
                #####

                # IoU
                if intersection.item() == 0 and union.item() == 0:
                    # Special case where we match an all-empty target
                    iou = np.nan
                else:
                    # Regular case
                    iou = (intersection / (union + eps)).item()

                self.results[f"#iou_{b}_{c}"] = iou
                bsl[f"#iou_{b}_{c}"] = (np.nan if np.isnan(iou) else bs * weight)
                #####

                # Precision
                if intersection.item() == 0 and p_cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    precision = np.nan
                else:
                    # Regular case
                    precision = (intersection / (p_cardinality + eps)).item()

                self.results[f"#precision_{b}_{c}"] = precision
                bsl[f"#precision_{b}_{c}"] = (np.nan if np.isnan(precision) else bs * weight)
                #####

                # Recall
                if intersection.item() == 0 and t_cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    recall = np.nan
                else:
                    # Regular case
                    recall = (intersection / (t_cardinality + eps)).item()

                self.results[f"#recall_{b}_{c}"] = recall
                bsl[f"#recall_{b}_{c}"] = (np.nan if np.isnan(recall) else bs * weight)
                #####

        # Compute the average for each metric. Exclude batches with edge cases (which were nan) from score and from seen.
        for c in range(nc):
            dice_seen = sum([v for b, v in bsl.items() if (b.startswith("#dice") and b.endswith(f"_{c}") and not np.isnan(v))])
            self.results[f"dice_avg_class_{c}"] = sum([self.results[f"#dice_{b}_{c}"] * bsl[f"#dice_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#dice_{b}_{c}"])]) / (dice_seen + eps)
            iou_seen = sum([v for b, v in bsl.items() if (b.startswith("#iou") and b.endswith(f"_{c}") and not np.isnan(v))])
            self.results[f"iou_avg_class_{c}"] = sum([self.results[f"#iou_{b}_{c}"] * bsl[f"#iou_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#iou_{b}_{c}"])]) / (iou_seen + eps)
            precision_seen = sum([v for b, v in bsl.items() if (b.startswith("#precision") and b.endswith(f"_{c}") and not np.isnan(v))])
            self.results[f"precision_avg_class_{c}"] = sum([self.results[f"#precision_{b}_{c}"] * bsl[f"#precision_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#precision_{b}_{c}"])]) / (precision_seen + eps)
            recall_seen = sum([v for b, v in bsl.items() if (b.startswith("#recall") and b.endswith(f"_{c}") and not np.isnan(v))])
            self.results[f"recall_avg_class_{c}"] = sum([self.results[f"#recall_{b}_{c}"] * bsl[f"#recall_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#recall_{b}_{c}"])]) / (recall_seen + eps)

        # Compute the unweighted average across all non-background (non-zero) classes
        if len(self.report_avg) > 0:
            self.results[f"dice_avg_overall"] = sum([self.results[f"dice_avg_class_{c}"] for c in range(nc) if c in self.report_avg]) / len(self.report_avg)
            self.results[f"iou_avg_overall"] = sum([self.results[f"iou_avg_class_{c}"] for c in range(nc) if c in self.report_avg]) / len(self.report_avg)
            self.results[f"precision_avg_overall"] = sum([self.results[f"precision_avg_class_{c}"] for c in range(nc) if c in self.report_avg]) / len(self.report_avg)
            self.results[f"recall_avg_overall"] = sum([self.results[f"recall_avg_class_{c}"] for c in range(nc) if c in self.report_avg]) / len(self.report_avg)

        return self.results
        
class BraTS_Metrics():

    """
    Same as Segmentation_Metrics, but computes against other targets, namely the constructed
    targets of ET, WT, and TC, because those are not provided in the BraTS segmentation data.
    """

    def __init__(self, classes, verbose: bool = False, **kwargs):
        self.results = OrderedDict()
        self.verbose = verbose
        self.kwargs = kwargs
        self.weights = self.kwargs.get("weights", None)
        self.classes = classes
        self.report_avg = self.kwargs.get("report_avg", [])
        self.metrics_for_background = self.kwargs.get("metrics_for_background", True)

    def forward(self, predictions: List[torch.Tensor, ], targets: List[torch.Tensor, ]):
        
        eps = 1e-6
        bsl = {}
        # De facto number of classes (including background computed at runtime)
        nc = (self.classes + 1 if self.metrics_for_background is True else self.classes)

        for b in range(len(targets)):
            
            # Convert predictions to binary one-hot format for proper scoring
            p_arg = nnf.one_hot(torch.argmax(predictions[b].to(torch.float32), dim = 1), num_classes = self.classes + 1).moveaxis(-1, 1)
            et_p = p_arg[:, 3, :, :] .to(dtype = torch.bool)
            tc_p = (p_arg[:, 3, :, :] + p_arg[:, 1, :, :]).to(dtype = torch.bool)
            wt_p = (p_arg[:, 3, :, :] + p_arg[:, 2, :, :] + p_arg[:, 1, :, :]).to(dtype = torch.bool)
            bg_p = p_arg[:, 0, :, :].to(dtype = torch.bool)
            mc_pred = torch.stack(tensors = [bg_p, et_p, wt_p, tc_p], dim = 1)

            et_t = targets[b][2].to(dtype = torch.bool)
            tc_t = (targets[b][2] + targets[b][0]).to(dtype = torch.bool)
            wt_t = (targets[b][2] + targets[b][1] + targets[b][0]).to(dtype = torch.bool)
            bg_t = ~wt_t
            mc_targets = torch.stack(tensors = [bg_t, et_t, wt_t, tc_t], dim = 1)

            # Iterate over all classes, with or without computed background
            for c in range(nc): 
                    
                target = mc_targets[:, c, :, :].to(dtype = torch.bool).squeeze()
                prediction = mc_pred[:, c, :, :].to(dtype = torch.bool)
                intersection = torch.sum(prediction * target)
                p_cardinality = torch.sum(prediction)
                t_cardinality = torch.sum(target)
                cardinality = p_cardinality + t_cardinality
                union = torch.sum((prediction + target))

                bs = target.size()[0]

                if self.weights is None:
                    weight = 1
                else:
                    weight = self.weights[c]

                # Dice Score
                if intersection.item() == 0 and cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    dice_score = np.nan
                else:
                    # Regular case
                    dice_score = (2. * intersection / (cardinality + eps)).item()

                self.results[f"#dice_{b}_{c}"] = dice_score
                bsl[f"#dice_{b}_{c}"] = (np.nan if np.isnan(dice_score) else bs * weight)
                #####

                # IoU
                if intersection.item() == 0 and union.item() == 0:
                    # Special case where we match an all-empty target
                    iou = np.nan
                else:
                    # Regular case
                    iou = (intersection / (union + eps)).item()

                self.results[f"#iou_{b}_{c}"] = iou
                bsl[f"#iou_{b}_{c}"] = (np.nan if np.isnan(iou) else bs * weight)
                #####

                # Precision
                if intersection.item() == 0 and p_cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    precision = np.nan
                else:
                    # Regular case
                    precision = (intersection / (p_cardinality + eps)).item()

                self.results[f"#precision_{b}_{c}"] = precision
                bsl[f"#precision_{b}_{c}"] = (np.nan if np.isnan(precision) else bs * weight)
                #####

                # Recall
                if intersection.item() == 0 and t_cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    recall = np.nan
                else:
                    # Regular case
                    recall = (intersection / (t_cardinality + eps)).item()

                self.results[f"#recall_{b}_{c}"] = recall
                bsl[f"#recall_{b}_{c}"] = (np.nan if np.isnan(recall) else bs * weight)
                #####

        # Compute the average for each metric. Exclude batches with edge cases (which were nan) from score and from seen.
        for c in range(nc):
            dice_seen = sum([v for b, v in bsl.items() if (b.startswith("#dice") and b.endswith(f"_{c}") and not np.isnan(v))])
            self.results[f"dice_avg_class_{c}"] = sum([self.results[f"#dice_{b}_{c}"] * bsl[f"#dice_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#dice_{b}_{c}"])]) / (dice_seen + eps)
            iou_seen = sum([v for b, v in bsl.items() if (b.startswith("#iou") and b.endswith(f"_{c}") and not np.isnan(v))])
            self.results[f"iou_avg_class_{c}"] = sum([self.results[f"#iou_{b}_{c}"] * bsl[f"#iou_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#iou_{b}_{c}"])]) / (iou_seen + eps)
            precision_seen = sum([v for b, v in bsl.items() if (b.startswith("#precision") and b.endswith(f"_{c}") and not np.isnan(v))])
            self.results[f"precision_avg_class_{c}"] = sum([self.results[f"#precision_{b}_{c}"] * bsl[f"#precision_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#precision_{b}_{c}"])]) / (precision_seen + eps)
            recall_seen = sum([v for b, v in bsl.items() if (b.startswith("#recall") and b.endswith(f"_{c}") and not np.isnan(v))])
            self.results[f"recall_avg_class_{c}"] = sum([self.results[f"#recall_{b}_{c}"] * bsl[f"#recall_{b}_{c}"] for b in range(len(targets)) if not np.isnan(self.results[f"#recall_{b}_{c}"])]) / (recall_seen + eps)

        # Compute the unweighted average across all non-background (non-zero) classes
        if len(self.report_avg) > 0:
            self.results[f"dice_avg_overall"] = sum([self.results[f"dice_avg_class_{c}"] for c in range(nc) if c in self.report_avg]) / len(self.report_avg)
            self.results[f"iou_avg_overall"] = sum([self.results[f"iou_avg_class_{c}"] for c in range(nc) if c in self.report_avg]) / len(self.report_avg)
            self.results[f"precision_avg_overall"] = sum([self.results[f"precision_avg_class_{c}"] for c in range(nc) if c in self.report_avg]) / len(self.report_avg)
            self.results[f"recall_avg_overall"] = sum([self.results[f"recall_avg_class_{c}"] for c in range(nc) if c in self.report_avg]) / len(self.report_avg)

        return self.results
        