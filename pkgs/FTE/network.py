import torch
import torch.nn as nn
import torchvision
from pkgs.backboned_unet.backboned_unet.unet import Unet
from typing import Callable, List

# ResNets
def get_resnet(name, pretrained: bool = False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet34": torchvision.models.resnet34(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
        "resnet101": torchvision.models.resnet101(pretrained=pretrained),
        "resnet152": torchvision.models.resnet152(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: torch.Tensor):
        return x

class ClassificationNetwork(nn.Module):
    """
    Classification networks are a normal ResNet, with pretrained encoder part and fresh head.
    """

    def __init__(self, enc_name: str, enc_weights, classes: int, loss_criterion: torch.nn.Module, tf: Callable = None, frozen_encoder: bool = False):
        super(ClassificationNetwork, self).__init__()

        if "resnet" in enc_name:
            self.encoder = get_resnet(enc_name, pretrained = False)
        else:
            raise NotImplementedError # TODO: Add a pretrained ViT
        
        self.encoder.load_state_dict(torch.load(enc_weights, map_location = torch.device("cpu")), strict = False)
        self.classes = classes
        self.n_features = self.encoder.fc.in_features
        self.encoder.fc = Identity()
        self.loss_criterion = loss_criterion
        self.tf = tf
        self.frozen_encoder = frozen_encoder

        if self.frozen_encoder is True:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

        # Get new head
        self.head = nn.Linear(self.n_features, self.classes, bias = False)

    def forward(self, data: torch.Tensor, target: torch.Tensor):
        if self.tf is not None and self.training is True:
            data, _ = self.tf(data)
        data = self.encoder(data)
        pred = self.head(data)
        loss = self.loss_criterion(pred, target)
        return pred, loss

class SegmentationNetwork(nn.Module):
    """
    Segmentation networks are UNets built from encoder backbones, using https://github.com/mkisantal/backboned-unet.
    """

    def __init__(
        self, 
        enc_name: str, 
        enc_weights, 
        classes: int, 
        loss_criterion: torch.nn.Module, 
        tf: Callable = None,
        **kwargs):

        super(SegmentationNetwork, self).__init__()

        self.unet = Unet(
            backbone_name = enc_name,
            classes = classes,
            custom_weights = enc_weights,
            **kwargs
        )
        self.add_module("unet", self.unet)
        self.loss_criterion = loss_criterion
        self.add_module("loss_criterion", self.loss_criterion)
        self.tf = tf

    def forward(self, data: torch.Tensor, targets: List[torch.Tensor, ]):
        if self.tf is not None and self.training is True:
            data, targets = self.tf(data, targets)
        pred = self.unet(data)
        loss = self.loss_criterion(pred, targets)
        return pred, loss