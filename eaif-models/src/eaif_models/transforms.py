import torchvision.transforms as T


vit_transforms = T.Compose(
    [
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


resnet_transforms = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

r3m_transforms = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),  # this divides by 255
        T.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 255, 1 / 255, 1 / 255]
        ),  # this will scale bact to [0-255]
    ]
)
