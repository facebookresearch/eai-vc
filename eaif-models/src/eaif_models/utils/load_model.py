import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import eaif_models.models.vit.vit_mae as models_mae


mae_transforms = T.Compose([
                        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

class MAE_embedding_model(torch.nn.Module):
    def __init__(self, checkpoint_path, arch='mae_vit_large_patch16'):
        super().__init__()
        # build model
        self.mae_model = getattr(models_mae, arch)()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.mae_model.load_state_dict(checkpoint['model'], strict=False)
    
    def forward(self, imgs, mask_ratio=0.0):
        latent, mask, ids_restore = self.mae_model.forward_encoder(imgs, mask_ratio)
        cls_latent = latent[:, 0, :]
        return cls_latent
