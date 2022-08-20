import torch
import torch.nn as nn
import numpy as np
import imageio

class Unflatten(nn.Module):
    def __init__(self, size):
        super(Unflatten, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(-1, *self.size)

class ResidualBlock(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(C, C, 3, stride=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(C, C, 3, stride=1, padding=1), nn.ReLU())
	
    def forward(self, x):
	    return x + self.layers(x)

# Model
class DecoderModel(torch.nn.Module):

    def __init__(self, r3m_dim=2048, mlp_dim=512, enc_dim=256):
        super().__init__()
        
        layers = [nn.Linear(r3m_dim, mlp_dim), nn.ELU(),
                  nn.Linear(mlp_dim, enc_dim), nn.ELU(),
                  nn.Linear(enc_dim, 128), nn.ELU(),
                  nn.Linear(128, 32*16*16), nn.ReLU(), Unflatten((32, 16, 16)),
                  nn.ConvTranspose2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
                  nn.ConvTranspose2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
                  ResidualBlock(128),
                  nn.ConvTranspose2d(128, 64, 3, stride=1, padding=1), nn.ReLU(),
                  ResidualBlock(64),
                  nn.ConvTranspose2d(64, 3, 3, stride=1, padding=1), nn.Sigmoid()]

        self.model = nn.Sequential(*layers) 

    def forward(self, input_tensor):

        return self.model(input_tensor)

    def save_gif(self, pred_imgs, save_str):
        pred_imgs = pred_imgs.cpu().detach().numpy().transpose(0,2,3,1)* 255.0 # [B, 64, 64, 3]
        pred_imgs = pred_imgs.astype(np.uint8)
        imageio.mimsave(save_str, pred_imgs)