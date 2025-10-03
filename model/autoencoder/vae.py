import torch
import torch.nn as nn
from autoencoder.encoder import Encoder
from autoencoder.decoder import Decoder

class VQVAE(nn.Module):
    def __init__(self, im_channels=3, num_down_layers=3, num_mid_layers=2, num_up_layers=3):
        super().__init__()
        self.encoder = Encoder(im_channels, num_down_layers, num_mid_layers)
        self.decoder = Decoder(im_channels, num_up_layers, num_mid_layers)
    
    def forward(self, x):
        z_e, quant_losses = self.encoder(x)  # Encode the input image to latent representation, and get quantization losses
        x_recon = self.decoder(z_e)  # Decode the latent representation back to image
        return x_recon, z_e, quant_losses