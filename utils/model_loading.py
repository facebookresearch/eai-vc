import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import os
import sys
from PIL import Image
from torchvision.transforms import InterpolationMode
from torch.nn.modules.linear import Identity
from r3m import load_r3m
import clip
sys.path.append('/home/aryanjain/representation_networks/mae')
import models_mae

CHECKPOINT_DIR = '/home/aryanjain/representation_networks/'

clip_vit_model, _clip_vit_preprocess = clip.load("ViT-B/32", device='cpu')
clip_rn50_model, _clip_rn50_preprocess = clip.load("RN50x16", device='cpu')

_resnet_transforms = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

_mae_transforms = T.Compose([
                        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

_r3m_transforms = T.Compose([
                        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                        T.CenterCrop(224),
                        T.ToTensor(),  # this divides by 255
                        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1/255, 1/255, 1/255]), # this will scale bact to [0-255]
                    ])


MODEL_LIST = ['random',
              'resnet50', 'resnet50_rand', 
              'clip_vit', 'clip_rn50',
              'mae_ViT-B', 'mae_ViT-L', 'mae_ViT-H',
              'moco_conv5', 'moco_conv4', 'moco_conv3',
              'moco_croponly_conv5', 'moco_croponly_conv4', 'moco_croponly_conv3',
              'fuse_moco_34', 'fuse_moco_35', 'fuse_moco_45', 'fuse_moco_345',
              'fuse_moco_croponly_34', 'fuse_moco_croponly_35', 'fuse_moco_croponly_45', 'fuse_moco_croponly_345',
              'moco_adroit', 'moco_kitchen', 'moco_dmc',
              # experimental ego4d stuff
              'r3m', 'moco_ego4d_100k', 'moco_ego4d_5m',
             ]


# ===================================
# Temporal Embedding Fusion
# ===================================
def fuse_embeddings_concat(embeddings: list):
    assert type(embeddings[0]) == np.ndarray
    return np.array(embeddings).ravel()

def fuse_embeddings_flare(embeddings: list):
    if type(embeddings[0]) == np.ndarray:
        history_window = len(embeddings)
        delta = [embeddings[i+1] - embeddings[i] for i in range(history_window-1)]
        delta.append(embeddings[-1].copy())
        return np.array(delta).ravel()
    elif type(embeddings[0]) == torch.Tensor:
        history_window = len(embeddings)
        # each embedding will be (Batch, Dim)
        delta = [embeddings[i+1] - embeddings[i] for i in range(history_window-1)]
        delta.append(embeddings[-1])
        return torch.cat(delta, dim=1)
    else:
        print("Unsupported embedding format in fuse_embeddings_flare.")
        print("Provide either numpy.ndarray or torch.Tensor.")
        quit()


# ===================================
# Model Loading
# ===================================
def load_pvr_model(embedding_name, seed = 123, input_type = np.ndarray, *args, **kwargs):
    
    assert embedding_name in MODEL_LIST
    
    # ============================================================
    # Random
    # ============================================================
    if embedding_name == 'random':
        # A small randomly initialized CNN, used for training from scratch baseline
        model = small_cnn(in_channels=3)
        embedding_dim, transforms = 1568, _resnet_transforms
    # ============================================================
    # ResNet50
    # ============================================================
    if embedding_name == 'resnet50':
        # ResNet50 pretrained on ImageNet
        model = models.resnet50(pretrained=True, progress=False)
        model.fc = Identity()
        embedding_dim, transforms = 2048, _resnet_transforms
    elif embedding_name == 'resnet50_rand':
        # Randomly initialized ResNet50 features
        model = models.resnet50(pretrained=False, progress=False)
        model.fc = Identity()
        embedding_dim, transforms = 2048, _resnet_transforms
    # ============================================================
    # MAE
    # ============================================================
    elif embedding_name == 'mae_ViT-B':
        model = MAE_embedding_model(checkpoint_path = CHECKPOINT_DIR + 'mae_pretrain_vit_base.pth', arch='mae_vit_base_patch16')
        embedding_dim = 768
        transforms = _mae_transforms
    elif embedding_name == 'mae_ViT-L':
        model = MAE_embedding_model(checkpoint_path = CHECKPOINT_DIR + 'mae_pretrain_vit_large.pth', arch='mae_vit_large_patch16')
        embedding_dim = 1024
        transforms = _mae_transforms
    elif embedding_name == 'mae_ViT-H':
        model = MAE_embedding_model(checkpoint_path = CHECKPOINT_DIR + 'mae_pretrain_vit_huge.pth', arch='mae_vit_huge_patch14')
        embedding_dim = 1280
        transforms = _mae_transforms
    # ============================================================
    # CLIP
    # ============================================================
    elif embedding_name == 'clip_vit':
        # CLIP with Vision Transformer architecture
        model = clip_vit_model.visual
        transforms = _clip_vit_preprocess
        embedding_dim = 512
    elif embedding_name == 'clip_rn50':
        # CLIP with ResNet50x16 (large model) architecture
        model = clip_rn50_model.visual
        transforms = _clip_rn50_preprocess
        embedding_dim = 768
    # ============================================================
    # MoCo (Aug+)
    # ============================================================
    elif embedding_name == 'moco_conv3':
        model, embedding_dim = moco_conv3_compression_model(CHECKPOINT_DIR + '/moco_v2_conv3.pth.tar')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_conv4':
        model, embedding_dim = moco_conv4_compression_model(CHECKPOINT_DIR + '/moco_v2_conv4.pth.tar')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_conv5':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_v2_800ep_pretrain.pth.tar')
        transforms = _resnet_transforms
    # ============================================================
    # MoCo (croponly)
    # ============================================================
    elif embedding_name == 'moco_croponly_conv3':
        model, embedding_dim = moco_conv3_compression_model(CHECKPOINT_DIR + '/moco_croponly_conv3.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_croponly_conv4':
        model, embedding_dim = moco_conv4_compression_model(CHECKPOINT_DIR + '/moco_croponly_conv4.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_croponly_conv5':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_croponly.pth')
        transforms = _resnet_transforms
    # ============================================================
    # MoCo (Aug+) multi-layer
    # ============================================================
    elif embedding_name == 'fuse_moco_34':
        m3, e3, t3 = load_pvr_model('moco_conv3', seed)
        m4, e4, t4 = load_pvr_model('moco_conv4', seed)
        model = CombinedModel([m3, m4])
        embedding_dim, transforms = e3+e4, _resnet_transforms
    elif embedding_name == 'fuse_moco_35':
        m3, e3, t3 = load_pvr_model('moco_conv3', seed)
        m5, e5, t5 = load_pvr_model('moco_conv5', seed)
        model = CombinedModel([m3, m5])
        embedding_dim, transforms = e3+e5, _resnet_transforms
    elif embedding_name == 'fuse_moco_45':
        m4, e4, t4 = load_pvr_model('moco_conv4', seed)
        m5, e5, t5 = load_pvr_model('moco_conv5', seed)
        model = CombinedModel([m4, m5])
        embedding_dim, transforms = e4+e5, _resnet_transforms
    elif embedding_name == 'fuse_moco_345':
        m3, e3, t3 = load_pvr_model('moco_conv3', seed)
        m4, e4, t4 = load_pvr_model('moco_conv4', seed)
        m5, e5, t5 = load_pvr_model('moco_conv5', seed)
        model = CombinedModel([m3, m4, m5])
        embedding_dim, transforms = e3+e4+e5, _resnet_transforms
    # ============================================================
    # MoCo (croponly) multi-layer
    # ============================================================
    elif embedding_name == 'fuse_moco_croponly_34':
        m3, e3, t3 = load_pvr_model('moco_croponly_conv3', seed)
        m4, e4, t4 = load_pvr_model('moco_croponly_conv4', seed)
        model = CombinedModel([m3, m4])
        embedding_dim, transforms = e3+e4, _resnet_transforms
    elif embedding_name == 'fuse_moco_croponly_35':
        m3, e3, t3 = load_pvr_model('moco_croponly_conv3', seed)
        m5, e5, t5 = load_pvr_model('moco_croponly_conv5', seed)
        model = CombinedModel([m3, m5])
        embedding_dim, transforms = e3+e5, _resnet_transforms
    elif embedding_name == 'fuse_moco_croponly_45':
        m4, e4, t4 = load_pvr_model('moco_croponly_conv4', seed)
        m5, e5, t5 = load_pvr_model('moco_croponly_conv5', seed)
        model = CombinedModel([m4, m5])
        embedding_dim, transforms = e4+e5, _resnet_transforms
    elif embedding_name == 'fuse_moco_croponly_345':
        m3, e3, t3 = load_pvr_model('moco_croponly_conv3', seed)
        m4, e4, t4 = load_pvr_model('moco_croponly_conv4', seed)
        m5, e5, t5 = load_pvr_model('moco_croponly_conv5', seed)
        model = CombinedModel([m3, m4, m5])
        embedding_dim, transforms = e3+e4+e5, _resnet_transforms
    # ============================================================
    # MoCo (aug+) trained on mujoco datasets
    # ============================================================
    elif embedding_name == 'moco_adroit':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_adroit.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_kitchen':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_kitchen.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_dmc':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_dmc.pth')
        transforms = _resnet_transforms
    # ============================================================
    # Ego4D models
    # ============================================================
    elif embedding_name == 'r3m':
        model = load_r3m("resnet50")
        model = model.module.eval()
        model = model.to('cpu')
        embedding_dim = 2048
        transforms = _r3m_transforms
    elif embedding_name == 'moco_ego4d_100k':
        # model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_ego4d_100k.pth')
        model, embedding_dim = moco_conv5_model('/checkpoint/aravraj/moco_diff_aug/ego4d_100k/checkpoints/ego4d_100k/checkpoint_0190.pth')
        transforms = _resnet_transforms
    elif embedding_name == 'moco_ego4d_5m':
        # model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_ego4d_5m.pth')
        model, embedding_dim = moco_conv5_model('/checkpoint/aravraj/moco_diff_aug/ego4d_5m/checkpoints/ego4d_5m/checkpoint_0010.pth')
        transforms = _resnet_transforms
    else:
        print("Model not implemented.")
        raise NotImplementedError
    model = model.eval()

    def final_transforms(transforms):
        if input_type == np.ndarray:
            return lambda input : transforms(Image.fromarray(input)).unsqueeze(0)
        else:
            return transforms
            
    return model, embedding_dim, final_transforms(transforms)


def moco_conv5_model(checkpoint_path):
    model = models.resnet50(pretrained=False, progress=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    model.fc = Identity()
    return model, 2048


def moco_conv4_compression_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    # construct the compressed model
    model = models.resnet.resnet50(pretrained=False, progress=False)
    downsample = nn.Sequential(
                    nn.Conv2d(2048,
                    42,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    dilation=1), model._norm_layer(42))
    model.layer4 = nn.Sequential(
                    model.layer4,
                    models.resnet.BasicBlock(2048,
                        42,
                        stride=1,
                        norm_layer=model._norm_layer,
                        downsample=downsample))
    # Remove the avgpool layer
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert all(['fc.' in n or 'layer4.2' in n  for n in msg.unexpected_keys])
    assert len(msg.missing_keys)==0
    # manually computed the embedding dimension to be 2058
    return model, 2058


def moco_conv3_compression_model(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    # construct the compressed model
    model = models.resnet.resnet50(pretrained=False, progress=False)
    downsample1 = nn.Sequential(
        nn.Conv2d(1024,
                  11,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  groups=1,
                  dilation=1), model._norm_layer(11))

    model.layer3 = nn.Sequential(
        model.layer3,
        models.resnet.BasicBlock(1024,
                                 11,
                                 stride=1,
                                 norm_layer=model._norm_layer,
                                 downsample=downsample1)
    )

    # Remove the avgpool layer
    model.layer4 = nn.Sequential()
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert all(['fc.' in n or 'layer4.' in n or 'layer3.2' in n for n in msg.unexpected_keys])
    assert len(msg.missing_keys)==0
    # manually computed the embedding dimension to be 2156
    return model, 2156


def small_cnn(in_channels=3):
    """
        Make a small CNN visual encoder
        Architecture based on DrQ-v2
    """
    model = nn.Sequential(nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),
                          nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),
                          nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), 
                          nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),
                          nn.ReLU(), nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32),
                          nn.ReLU(), nn.Flatten())
    return model


class CombinedModel(nn.Module):
    def __init__(self, 
                 model_list: list = None,
                 ):
        """
            Combines features (outputs) from multiple models.
        """
        super(CombinedModel, self).__init__()
        self.models = model_list
        
    def to(self, device):
        for idx in range(len(self.models)):
            self.models[idx] = self.models[idx].to(device)
        return super().to(device)

    def forward(self, x):
        layer_outs = [model.forward(x) for model in self.models]
        return torch.cat(layer_outs, axis=-1)


class MAE_embedding_model(torch.nn.Module):
    def __init__(self, checkpoint_path, arch='mae_vit_large_patch16'):
        super().__init__()
        # build model
        model = getattr(models_mae, arch)()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        # print(msg)
        self.mae_model = model
    
    def forward(self, imgs, mask_ratio=0.0):
        latent, mask, ids_restore = self.mae_model.forward_encoder(imgs, mask_ratio)
        cls_latent = latent[:, 0, :]
        return cls_latent