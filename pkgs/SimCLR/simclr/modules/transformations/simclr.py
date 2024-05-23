import torch, torchvision


class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly
    resulting in two correlated views of the same example,
    denoted x ̃i and x ̃j, which we consider as a positive pair.
    """

    def __init__(self, size, **kwargs):
        to_tensor = kwargs.get("to_tensor", False)
        grayscale = kwargs.get("grayscale", False)

        s = 1
        color_jitter = torchvision.transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        transforms = [
            torchvision.transforms.RandomResizedCrop(size=size),
            torchvision.transforms.RandomHorizontalFlip(),  # with 0.5 probability
            torchvision.transforms.RandomApply([color_jitter], p=0.8)
                    ]
        if grayscale != True:
            transforms.append(torchvision.transforms.RandomGrayscale(p=0.2))
        if to_tensor == True:
            transforms.append(torchvision.transforms.ToTensor())

        self.train_transform = torchvision.transforms.Compose(transforms)

        self.test_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize(size=size),
                torchvision.transforms.ToTensor(),
            ]
        )

    def __call__(self, x):
        #return x, x
        #if torch.any(torch.isnan(x)).item() == True or torch.any(torch.isinf(x)).item() == True:
        #    print("NAN/INF")
        #else:
        #    print("nada")
        return self.train_transform(x), self.train_transform(x)

