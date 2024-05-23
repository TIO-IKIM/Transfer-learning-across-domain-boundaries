import torch
import torch.nn as nn
import torchvision

try:
    from simclr.modules.resnet_hacks import modify_resnet_model
    from simclr.modules.identity import Identity
    from simclr.modules import NT_Xent
except:
    from pkgs.SimCLR.simclr.modules.resnet_hacks import modify_resnet_model
    from pkgs.SimCLR.simclr.modules.identity import Identity
    from pkgs.SimCLR.simclr.modules import NT_Xent

class SimCLR(nn.Module):
    """
    We opt for simplicity and adopt the commonly used ResNet (He et al., 2016) to obtain hi = f(x ̃i) = ResNet(x ̃i) where hi ∈ Rd is the output after the average pooling layer.
    """

    def __init__(self, encoder, args, n_features):
        super(SimCLR, self).__init__()

        self.encoder = encoder
        self.n_features = n_features
        self.loss_device = args.loss_device
        if not args.batch_size % args.gpus_dp == 0:
            raise ValueError("Batch size must be divisible by available GPUs.")
        if self.loss_device == "gpu":
            self.ntxent = NT_Xent(int(args.batch_size/args.gpus_dp), args.temperature, 1)
            self.add_module("ntxent", self.ntxent)

        # Replace the fc layer with an Identity function
        if "resnet" in args.encoder_network:
            self.encoder.fc = Identity()
        elif "vit" in args.encoder_network:
            self.encoder.heads = Identity()
        else:
            raise NotImplementedError

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.Sigmoid(),
            nn.Linear(self.n_features, args.projection_dim, bias=False)
        )

    def loss_calc(self, z_i, z_j):
        loss  = self.ntxent(z_i, z_j)
        return loss

    def forward(self, x, tf=None, device="cuda"):
        if tf is None:
            if isinstance(x, list):
                x_i, x_j = x[0].to(device), x[1].to(device)
            else:
                x_i, x_j = x.to(device)
        else:
            (x_i, x_j), _ = tf(x.to(device))
            del(x)

        #print("x_i:", torch.any(torch.isnan(x_i) + torch.isinf(x_i)))
        #print("x_j:", torch.any(torch.isnan(x_j) + torch.isinf(x_j)))

        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        #if (torch.any(torch.isnan(h_i) + torch.isinf(h_i)) or 
        #    torch.any(torch.isnan(h_j) + torch.isinf(h_j)) == True):
        #    for name, param in self.encoder.named_parameters():
        #        if torch.any(torch.isnan(param) + torch.isinf(param)) == True:
        #            print(f"{name} has nan/inf in param.")
        #        else:
        #            print(f"{name} is clear.")
        #print("h_i:", torch.any(torch.isnan(h_i) + torch.isinf(h_i)))
        #print("h_j:", torch.any(torch.isnan(h_j) + torch.isinf(h_j)))

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)

        #print("z_i:", torch.any(torch.isnan(z_i) + torch.isinf(z_i)))
        #print("z_j:", torch.any(torch.isnan(z_j) + torch.isinf(z_j)))

        if self.loss_device == "gpu":
            #print(z_i.size(), z_j.size())
            loss = self.loss_calc(z_i, z_j)
            return loss
        elif self.loss_device == "cpu":
            return z_i, z_j
        else:
            raise ValueError("loss_device must be cpu or gpu.")
