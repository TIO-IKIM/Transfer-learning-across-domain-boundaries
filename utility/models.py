import sys, os
sys.path.append("./")

import torch, torch.nn as nn, torchvision
import pkgs.SimCLR.simclr.modules.sync_batchnorm.batchnorm as sbn
from pkgs.backboned_unet.backboned_unet.unet import Unet

# Will hopefully be extended at some point
valid_models = [
    "resnet50",
    "unet_resnet50_backbone"
]

# All classes are the same as used in the framework, but without transformations or loss inside the model
class Classifier(nn.Module):
    """
    Classification networks are a normal ResNet, with pretrained encoder part and fresh head.
    This class is the same as the one used in the framework, except transformations and loss
    functions are not inside the model.


    """

    def __init__(
            self, 
            backbone_name: str,
            enc_weights: str, 
            classes: int, 
            frozen_encoder: bool = False
            ):
        super(Classifier, self).__init__()

        self.classes = classes

        if backbone_name == "resnet50":
            backbone = torchvision.models.resnet50()
        else:
            raise NotImplementedError
        self.encoder = sbn.convert_model(backbone)
        self.n_features = self.encoder.fc.in_features
        self.encoder.fc = torch.nn.Identity()
        self.head = nn.Linear(self.n_features, self.classes, bias = False)
        
        # If the model is one of the finetuned ones, every module has 'encoder' in its name already
        if "_FT_" in enc_weights:
            self.load_state_dict(torch.load(enc_weights, map_location = torch.device("cpu")), strict = True)
        # If its a pretrained one, we load into the encoder, so the layer names match up
        else:
            self.encoder.load_state_dict(torch.load(enc_weights, map_location = torch.device("cpu")), strict = True)

        self.frozen_encoder = frozen_encoder
        if self.frozen_encoder is True:
            for name, param in self.encoder.named_parameters():
                param.requires_grad = False

    def forward(self, data: torch.Tensor):
        data = self.encoder(data)
        return self.head(data)

class Segmenter(nn.Module):
    """
    Segmentation networks are UNets built from encoder backbones, using https://github.com/mkisantal/backboned-unet.
    """

    def __init__(
        self, 
        backbone_name: str, 
        enc_weights: str, 
        classes: int,
        frozen_encoder: bool = False):

        super(Segmenter, self).__init__()

        # load state dict
        state_dict = torch.load(enc_weights, map_location = torch.device("cpu"))
        
        # if we want a finetuned model, build as normal, load entire state_dict at once
        if "_FT_" in enc_weights:
            # so strict=True loading doesn't break
            del(state_dict["loss_criterion.XE.weight"])
            # build unet
            self.unet = Unet(
            backbone_name = backbone_name,
            classes = classes,
            custom_weights = None,
            encoder_freeze = frozen_encoder
            )
            # use state dict
            self.load_state_dict(state_dict)
            for (name1, _), (name2, _2) in zip(self.unet.named_parameters(), state_dict.items()):
                print(name1, name2)
        # else load the state_dict as we would during finetuning, by handing the weights at unet construction
        else:
            self.unet = Unet(
            backbone_name = backbone_name,
            classes = classes,
            custom_weights = enc_weights,
            encoder_freeze = frozen_encoder
            )
        # finally, add module
        self.add_module("unet", self.unet)

    def forward(self, data: torch.Tensor):
        return self.unet(data)

def get_model(
        model_name: str, 
        enc_weights: str, 
        classes: int,
        frozen_encoder: bool = False,
        ):
    
    """
    This function makes the models from the paper available in one line, without needing to frak about with the framework itself.
    
    Input:  model_name: str - Name of the model to build. Must be in valid_models.
            enc_weights: str - Location from which to load the model state dict.
            classes: int - The number of output classes in the model head. If you want to predict background as well, background should be included in the numnber of classes.
            frozen_encoder: bool - If True (default is False), all parameters except those in self.head are frozen.
    Output: A regular PyTorch model with a forward pass. This model does not do transformations or loss calculation, whereas the ones in the framework do.
    """

    if model_name == "resnet50":
            
        model = Classifier(
            backbone_name = "resnet50",
            enc_weights = enc_weights, 
            classes = classes, 
            frozen_encoder = frozen_encoder
            )
        
        return model
    
    elif model_name == "unet_resnet50_backbone":
    
        model = Segmenter(
            backbone_name = "resnet50",
            classes = classes,
            enc_weights = enc_weights,
            frozen_encoder = frozen_encoder
        )

        return model

    else:
        raise NotImplementedError(f"Model {model_name} not implemented. Model name must be one of {valid_models}.")

if __name__ == "__main__":
    #pt_resnet = get_model("resnet50", "./logs_and_checkpoints/pretraining/sancheck_1k100/encoder_pretrained.tar", 1000)
    #ft_resnet = get_model("resnet50", "./logs_and_checkpoints/finetuning/E1/PT_SimCLR_I1k_FT_I1k/finetuned_model.tar", 1000)
    #pt_unet = get_model("unet_resnet50_backbone", "./logs_and_checkpoints/pretraining/sancheck_ct100/encoder_pretrained.tar", 3)
    ft_unet = get_model("unet_resnet50_backbone", "./logs_and_checkpoints/finetuning/E1/PT_SimCLR_R_FT_LiTS/finetuned_model.tar", 3)
    #print(pt_resnet)
    #print(ft_resnet)
    #print(pt_unet)
    print(ft_unet)