import torch, torchvision
import utility.utils as uu
import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations import random_utils
from albumentations.augmentations.functional import clipped
import numpy as np
import cv2
import random

class GaussianNoise(torch.nn.Module):

    """
    Add random gaussian noise to your tensor. If p == 0., the tensor is simply passed.
    If clamp01 is True, the tensor is clamped to between 0 and 1. This is NOT done if p == 0.
    sig is SIGMA, NOT THE VARIANCE.
    
    Note that this augmentation does not integrate with albumentations.
    """

    def __init__(self, mu: float = 0, sig: float = 0.1, p: float = 0.2, clamp01 = True):
        self.mu = torch.tensor(mu, dtype = torch.float32)
        self.sig = torch.tensor(sig, dtype = torch.float32)
        self.p = p
        self.clamp01 = clamp01

    def __call__(self, x):
        if self.p == 0.:
            return x
        if torch.rand(1) <= self.p:
            x = x + (torch.randn_like(x) * self.sig) + self.mu # DEBUG
            if self.clamp01 is True:
                x = torch.clamp(x, 0., 1.)
            return x
        else:
            return x

class HalfPosterize(torch.nn.Module):

    def __init__(self, bits = 2, p = 0.2):
        self.bits = bits
        self.p = p

    def __call__(self, x):
        if "float" in str(x.dtype):
            raise TypeError("HalfPosterize should only be used on int-type images, as it works differently on floats.")
        bytes = x.dtype.nbytes
        for b in range(self.bits):
            uu.set_bit(x, bytes-b-1, 0)
        return x

class RicianNoise(ImageOnlyTransform):
    """
    Apply random Rician noise to the input image.

    Everything is like the regular albumentations GaussNoise, except the noise is Rician.
    """

    def __init__(
            self, 
            var_limit = (10.0, 50.0),
            mean = 0, 
            per_channel = True,
            always_apply = False, 
            p = 0.5):
        
        super(RicianNoise, self).__init__(always_apply, p)
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )

        self.mean = mean
        self.per_channel = per_channel

    def apply(self, img, noise1 = None, noise2 = None, **params):
        return self.add_rician_noise(img, noise1, noise2)
    
    @staticmethod
    @clipped
    def add_rician_noise(img, noise1, noise2):
        img = img.astype("float32")
        return np.sqrt((img + noise1)**2 + noise2**2)
    
    def get_params_dependent_on_targets(self, params):
        image = params["image"]
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var**0.5

        if self.per_channel:
            #gauss = random_utils.normal(self.mean, sigma, image.shape)
            noise1 = random_utils.normal(self.mean, sigma, size = image.shape).astype(image.dtype, copy=False)
            noise2 = random_utils.normal(self.mean, sigma, size = image.shape).astype(image.dtype, copy=False)
        else:
            #gauss = random_utils.normal(self.mean, sigma, image.shape[:2])
            noise1 = random_utils.normal(self.mean, sigma, size = image.shape[:2]).astype(image.dtype, copy=False)
            noise2 = random_utils.normal(self.mean, sigma, size = image.shape[:2]).astype(image.dtype, copy=False)
            if len(image.shape) == 3:
                noise1 = np.expand_dims(noise1, -1)
                noise2 = np.expand_dims(noise2, -1)

        return {"noise1": noise1, "noise2": noise2}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return ("var_limit", "per_channel", "mean")

def ImageNet_augs(noise_injection = None):

    cpu_augs = None
    gpu_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomApply([torchvision.transforms.RandomResizedCrop((256, 256), antialias = False)], 0.2),
    torchvision.transforms.RandomHorizontalFlip(p = 0.2),
    #torchvision.transforms.RandomVerticalFlip(p = 0.2),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.2)], p = 0.2),
    torchvision.transforms.RandomGrayscale(p = 0.2),
    #torchvision.transforms.RandomSolarize(0.8, p = 0.2),
    torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(3, (0.1, 2))], p = 0.2),
    torchvision.transforms.RandomAdjustSharpness(2, p = 0.2),
    GaussianNoise(
        mu = 0, 
        sig = (1e-3 if noise_injection is None else noise_injection[0]),
        p = (0. if noise_injection is None else noise_injection[1])
        ),
    ])

    return cpu_augs, gpu_augs

def CTBR_augs(noise_injection = None):

    cpu_augs = None
    gpu_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomApply([torchvision.transforms.RandomResizedCrop((256, 256), antialias = False)], 0.2),
    torchvision.transforms.RandomHorizontalFlip(p = 0.2),
    #torchvision.transforms.RandomVerticalFlip(p = 0.2),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.2)], p = 0.2),
    #torchvision.transforms.RandomGrayscale(p = 0.2),
    #torchvision.transforms.RandomSolarize(0.8, p = 0.2),
    torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(3, (0.1, 2))], p = 0.2),
    torchvision.transforms.RandomAdjustSharpness(2, p = 0.2),
    GaussianNoise(
        mu = 0, 
        sig = (1e-3 if noise_injection is None else noise_injection[0]),
        p = (0. if noise_injection is None else noise_injection[1])
        ),
    ])

    return cpu_augs, gpu_augs

def LiTS_augs(noise_injection = None):

    cpu_augs = None
    gpu_augs = albumentations.Compose([
        albumentations.RandomResizedCrop(height = 256, width = 256, p = 0.2),
        albumentations.HorizontalFlip(p = 0.2),
        #albumentations.VerticalFlip(p = 0.2),
        albumentations.ColorJitter(0.1, 0.1, 0.1, 0.2, p = 0.2),
        #albumentations.ToGray(p = 0.2),
        #albumentations.Solarize(0.8, p = 0.2),
        albumentations.GaussianBlur((3, 3), (0.1, 2.), p = 0.2),
        #albumentations.Sharpen(alpha = (0.33, 0.33), lightness = (0.5, 0.5), p = 1.0),
        albumentations.GaussNoise(
            mean = 0, 
            var_limit = ((1e-6, 1e-6) if noise_injection is None else ((noise_injection[0])**2, (noise_injection[0])**2)), 
            p = (0. if noise_injection is None else noise_injection[1])
            ),
    ])

    return cpu_augs, gpu_augs

def BraTS_augs(noise_injection = None):

    cpu_augs = None
    gpu_augs = albumentations.Compose([
        albumentations.RandomResizedCrop(height = 256, width = 256, p = 0.2),
        albumentations.HorizontalFlip(p = 0.2),
        #albumentations.VerticalFlip(p = 0.2),
        #albumentations.ColorJitter(0.1, 0.1, 0.1, 0.2, p = 0.2),
        #albumentations.ToGray(p = 0.2),
        #albumentations.Solarize(0.8, p = 0.2),
        albumentations.GaussianBlur((3, 3), (0.1, 2.), p = 0.2),
        #albumentations.Sharpen(alpha = (0.33, 0.33), lightness = (0.5, 0.5), p = 1.0),
        RicianNoise(
            mean = 0, 
            var_limit = ((1e-6, 1e-6) if noise_injection is None else ((noise_injection[0])**2, (noise_injection[0])**2)), 
            p = (0. if noise_injection is None else noise_injection[1])
            ),
    ])

    return cpu_augs, gpu_augs

"""
def PASCAL_VOC_augs(noise_injection = None):

    cpu_augs = None
    gpu_augs = albumentations.Compose([
        albumentations.RandomResizedCrop(height = 256, width = 256, p = 0.2, interpolation = cv2.INTER_NEAREST),
        albumentations.HorizontalFlip(p = 0.2),
        #albumentations.VerticalFlip(p = 0.2),
        albumentations.ColorJitter(0.1, 0.1, 0.1, 0.2, p = 0.2),
        albumentations.ToGray(p = 0.2),
        #albumentations.Solarize(0.8, p = 0.2),
        albumentations.GaussianBlur((3, 3), (0.1, 2.), p = 0.2),
        #albumentations.Sharpen(alpha = (0.33, 0.33), lightness = (0.5, 0.5), p = 0.2),
        albumentations.GaussNoise(
            mean = 0, 
            var_limit = ((1e-6, 1e-6) if noise_injection is None else ((noise_injection[0])**2, (noise_injection[0])**2)), 
            p = (0. if noise_injection is None else noise_injection[1])
            ),
    ])

    return cpu_augs, gpu_augs
"""

def CX8_augs(noise_injection = None):

    cpu_augs = None
    gpu_augs = torchvision.transforms.Compose([
    torchvision.transforms.RandomApply([torchvision.transforms.RandomResizedCrop((256, 256), antialias = False)], 0.2),
    torchvision.transforms.RandomHorizontalFlip(p = 0.2),
    #torchvision.transforms.RandomVerticalFlip(p = 0.2),
    torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.1, 0.1, 0.1, 0.2)], p = 0.2),
    #torchvision.transforms.RandomGrayscale(p = 0.2),
    #torchvision.transforms.RandomSolarize(0.8, p = 0.2),
    torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(3, (0.1, 2))], p = 0.2),
    torchvision.transforms.RandomAdjustSharpness(2, p = 0.2),
    GaussianNoise(
        mu = 0, 
        sig = (1e-3 if noise_injection is None else noise_injection[0]),
        p = (0. if noise_injection is None else noise_injection[1])
        ),
    ])

    return cpu_augs, gpu_augs
