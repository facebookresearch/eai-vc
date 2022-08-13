import torch
import torchvision.models as models, torchvision.transforms as T

_resnet_transforms = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

def load_model(version='resnet50', imagenet_pretrained=True, metadata=None):
    """
    Load a resnet model with random weights, imagenet pretrained weights, or from a checkpoint.
    """
    assert version in ['resnet50', 'resnet18', 'resnet34']
    if version == 'resnet50':
        model = models.resnet50(pretrained=imagenet_pretrained, progress=False)
    elif version == 'resnet34':
        model = models.resnet34(pretrained=imagenet_pretrained, progress=False)
    elif version == 'resnet18':
        model = models.resnet18(pretrained=imagenet_pretrained, progress=False)
    model.fc = torch.nn.modules.linear.Identity()
    embedding_dim, transforms = 2048, _resnet_transforms
    return model, embedding_dim, transforms, metadata
