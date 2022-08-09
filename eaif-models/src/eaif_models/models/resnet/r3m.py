import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

from r3m import load_r3m


_r3m_transforms = T.Compose([
                        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                        T.CenterCrop(224),
                        T.ToTensor(),  # this divides by 255
                        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1/255, 1/255, 1/255]), # this will scale bact to [0-255]
                    ])

def load_model():
    model = load_r3m("resnet50")
    model = model.module.eval()
    model = model.to('cpu')
    embedding_dim = 2048
    transforms = _r3m_transforms

    return model, embedding_dim, transforms