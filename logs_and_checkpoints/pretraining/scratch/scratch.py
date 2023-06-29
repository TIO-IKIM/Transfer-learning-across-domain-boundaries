import torch, torchvision
import sys, os
sys.path.append("/Projects/DATL")
from pkgs.SimCLR.simclr import SimCLR
import utility.utils as uu

args = uu.arg()
args.batch_size = 4096
args.loss_device = "cpu"
args.gpus_dp = 8
args.temperature = 0.5
args.projection_dim = 256

encoder = torchvision.models.resnet152(pretrained=False)
model = SimCLR(encoder, args, encoder.fc.in_features)
out = os.path.join("/Projects/DATL/logs_and_checkpoints/pretraining/scratch", "scratch_resnet_152.tar")
torch.save(model.encoder.state_dict(), out)