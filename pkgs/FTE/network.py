import torch
import torch.nn as nn
import torchvision

#from einops import rearrange
from pkgs.backboned_unet.backboned_unet.unet import Unet
#from pkgs.MAE.model import MAE_ViT
from typing import Callable, List

# ResNets
def get_resnet(name, pretrained: bool = False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained = pretrained),
        "resnet34": torchvision.models.resnet34(pretrained = pretrained),
        "resnet50": torchvision.models.resnet50(pretrained = pretrained),
        "resnet101": torchvision.models.resnet101(pretrained = pretrained),
        "resnet152": torchvision.models.resnet152(pretrained = pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]

#def get_vit(name, pretrained: bool = False):
#    vits = {
#        "mae_vit": MAE_ViT(
#            image_size = 224,
#            patch_size = 16,
#            emb_dim = 768,
#            encoder_layer = 12,
#            encoder_head = 6,
#            decoder_layer = 12,
#            decoder_head = 6,
#            mask_ratio = 0.75
#            )
#    }
#    if name not in vits.keys():
#        raise KeyError(f"{name} is not a tested model")
#    return vits[name]

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

        self.enc_name = enc_name
        self.classes = classes
        sd = torch.load(enc_weights, map_location = torch.device("cpu"))

        if "resnet" in enc_name:
            self.encoder = get_resnet(enc_name, pretrained = False)
            self.encoder.load_state_dict(sd, strict = False)
        #elif "vit" in enc_name:
        #    vit_model = get_vit(enc_name, pretrained = False)
        #    vit_model.load_state_dict(sd, strict = True)
        #    self.encoder = vit_model.encoder
        
        if "resnet" in enc_name:
            self.n_features = self.encoder.fc.in_features
            self.encoder.fc = Identity()
            self.head = nn.Linear(self.n_features, self.classes, bias = False)
        #elif "vit" in enc_name:
        #    self.cls_token = self.encoder.cls_token
        #    self.pos_embedding = self.encoder.pos_embedding
        #    self.patchify = self.encoder.patchify
        #    self.transformer = self.encoder.transformer
        #    self.layer_norm = self.encoder.layer_norm
        #    self.head = torch.nn.Linear(self.pos_embedding.shape[-1], self.classes)

        self.loss_criterion = loss_criterion
        self.tf = tf
        self.frozen_encoder = frozen_encoder

        if self.frozen_encoder is True:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(self, data: torch.Tensor, target: torch.Tensor):
        if "resnet" in self.enc_name:
            if self.tf is not None and self.training is True:
                data, _ = self.tf(data)
            data = self.encoder(data)
            pred = self.head(data)
            loss = self.loss_criterion(pred, target)
            return pred, loss
        
        #elif "vit" in self.enc_name:
        #    if self.tf is not None and self.training is True:
        #        data, _ = self.tf(data)
        #    patches = self.patchify(data)
        #    patches = rearrange(patches, 'b c h w -> (h w) b c')
        #    patches = patches + self.pos_embedding
        #    patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        #    patches = rearrange(patches, 't b c -> b t c')
        #    features = self.layer_norm(self.transformer(patches))
        #    features = rearrange(features, 'b t c -> t b c')
        #    pred = self.head(features[0])
        #    loss = self.loss_criterion(pred, target)
        #    return pred, loss

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