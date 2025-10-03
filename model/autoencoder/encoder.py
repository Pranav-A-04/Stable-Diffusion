import torch
import torch.nn as nn
from utils.blocks import DownSampleBlock, MidBlock

class Encoder(nn.Module):
    def __init__(self, im_channels, num_down_layers, num_mid_layers):
        super().__init__()
        self.down_channels = [64, 128, 256, 256]
        self.mid_channels = [256, 256]
        self.down_sample = [True, True, True]
        self.down_sample_attention = [False, False, False] # no attention in downsample blocks
        self.num_heads = 4
        self.latent_dim = 3
        self.norm_channels = 32
        
        # need smthn to get the im_channels in down_channels[0]
        self.encoder_conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        # encoder down blocks
        self.encoder_down_blocks = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            assert self.down_channels[i] <= self.down_channels[i+1], "Down channels should be in non-decreasing order"
            self.encoder_down_blocks.append(
                DownSampleBlock(
                    in_channels=self.down_channels[i],
                    out_channels=self.down_channels[i+1],
                    down_sample=self.down_sample[i],
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels,
                    t_emb_dim=None,
                    self_attention=self.down_sample_attention[i]
                )
            )
        
        # encoder mid blocks
        self.encoder_mid_blocks = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):           
            self.encoder_mid_blocks.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i+1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels
                )
            )
        
        # codebook - for VQ-VAE
        self.codebook_size = 8192
        self.encoder_norm_out = nn.GroupNorm(self.norm_channels, self.down_channels[-1])
        self.encoder_conv_out = nn.Conv2d(self.down_channels[-1], self.latent_dim, kernel_size=3, padding=1)
        
        # pre quant conv
        self.encoder_pre_quant_conv = nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=1)
        
        # embedding layer for codebook
        self.codebook = nn.Embedding(self.codebook_size, self.latent_dim)
    
    def quantize(self, x):
        B, C, H, W = x.shape
        
        # permute to (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # reshape to (B, H*W, C)
        x = x.reshape(B, H * W, C)
        
        # compute the distances against all codebook vectors
        # x has shape (B, H*W, C)
        # codebook.weight has shape (codebook_size(K), C)
        # First we make our weights -> (1, K, C) by unsqueezing at dim 0
        # Then repeat along dim 0 for B times -> (B, K, C)
        # Now pairwise dist -> (B, H*W, K)
        distances = torch.cdist(x, self.codebook.weight.unsqueeze(0).repeat(B, 1, 1))
        min_dist_indices = torch.argmin(distances, dim=-1) # (B, H*W)
        
        # get quantized vectors from codebook
        quant_out = torch.index_select(self.codebook.weight, 0, min_dist_indices.view(-1)) # (B*H*W, C)
        
        x = x.reshape((-1, C))
        commitment_loss = torch.mean((quant_out.detach() - x) ** 2)
        codebook_loss = torch.mean((quant_out - x.detach()) ** 2)
        
        quant_losses = {
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss
        }
        
        quant_out = x + (quant_out - x).detach()
        
        quant_out = quant_out.view(B, H, W, C).permute(0, 3, 1, 2) # (B, C, H, W)
        return quant_out, quant_losses
        
        
        
    def forward(self, x):
        out = self.encoder_conv_in(x)
        
        # down blocks
        for down in self.encoder_down_blocks:
            out = down(out, t_emb=None)
        
        # mid blocks
        for mid in self.encoder_mid_blocks:
            out = mid(out, t_emb=None)
        
        out = self.encoder_norm_out(out)
        out = nn.SiLU(out)
        out = self.encoder_conv_out(out)
        
        # pre quant conv
        out = self.encoder_pre_quant_conv(out)
        quant_out, quant_losses = self.quantize(out)
        
        return quant_out, quant_losses
        
        
        
        
        
        
        