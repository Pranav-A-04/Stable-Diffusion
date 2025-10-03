import torch
import torch.nn as nn
from utils.blocks import UpSampleBlock, MidBlock

class Decoder(nn.Module):
    def __init__(self, im_channels, num_up_layers, num_mid_layers):
        super().__init__()
        self.up_channels = [128, 64, 32]
        self.mid_channels = [128, 128]
        self.up_sample = [True, True, True]
        self.up_sample_attention = [False, False, False] # no attention in upsample blocks
        self.num_heads = 16
        self.latent_dim = 3
        self.norm_channels = 32
        
        # need a conv post quantization
        self.post_quant_conv = nn.Conv2d(self.latent_dim, self.latent_dim, kernel_size=1)
        
        # need smthn to get the latent_dim in up_channels[0]
        self.decoder_conv_in = nn.Conv2d(self.latent_dim, self.mid_channels[-1], kernel_size=3, padding=1)
        
        # decoder mid blocks
        self.decoder_mid_blocks = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):           
            self.decoder_mid_blocks.append(
                MidBlock(
                    in_channels=self.mid_channels[i],
                    out_channels=self.mid_channels[i+1],
                    t_emb_dim=None,
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels
                )
            )
        
        # decoder up blocks
        self.decoder_up_blocks = nn.ModuleList([])
        for i in range(len(self.up_channels)-1):
            assert self.up_channels[i] >= self.up_channels[i+1], "Up channels should be in non-increasing order"
            self.decoder_up_blocks.append(
                UpSampleBlock(
                    in_channels=self.up_channels[i],
                    out_channels=self.up_channels[i+1],
                    up_sample=self.up_sample[i],
                    num_heads=self.num_heads,
                    norm_channels=self.norm_channels,
                    t_emb_dim=None,
                    self_attention=self.up_sample_attention[i]
                )
            )
        
        # need smthn to get the up_channels[-1] to im_channels
        self.decoder_norm_out = nn.GroupNorm(self.norm_channels, self.up_channels[-1])
        self.decoder_conv_out = nn.Conv2d(self.up_channels[-1], im_channels, kernel_size=3, padding=1)
    
    def forward(self, z):
        out = z
        out = self.post_quant_conv(out)
        out = self.decoder_conv_in(out)
        
        #mid blocks
        for mid in self.decoder_mid_blocks:
            out = mid(out, t_emb=None)
        
        #upsample blocks
        for up in self.decoder_up_blocks:
            out = up(out, None, t_emb=None)
        
        out = self.decoder_norm_out(out)
        out = nn.SiLU()(out)
        out = self.decoder_conv_out(out)
        return out