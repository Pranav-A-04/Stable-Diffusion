import torch
import torch.nn as nn
from utils.blocks import DownSampleBlock, UpSampleBlock, MidBlock
from utils.misc import get_time_embedding

class Unet(nn.Module):
    def __init__(self, im_channels):
        super().__init__()
        self.down_channels = [32, 64, 128, 256]
        self.mid_channels = [256, 256, 128]
        self.t_emb_dim = 128
        self.down_sample = [True, True, False]
        
        self.t_proj = nn.Sequential(
            nn.Linear(self.t_emb_dim, self.t_emb_dim),
            nn.SiLU(),
            nn.Linear(self.t_emb_dim, self.t_emb_dim)
        )
        self.up_sample = list(reversed(self.down_sample))
        self.conv_in = nn.Conv2d(im_channels, self.down_channels[0], kernel_size=3, padding=1)
        
        self.downs = nn.ModuleList([])
        for i in range(len(self.down_channels)-1):
            self.downs.append(DownSampleBlock(
                self.down_channels[i], 
                self.down_channels[i+1], 
                self.t_emb_dim, 
                self.down_sample[i], 
                num_heads=4
                )
            )
            
        self.mids = nn.ModuleList([])
        for i in range(len(self.mid_channels)-1):
            self.mids.append(MidBlock(self.mid_channels[i], self.mid_channels[i+1], self.t_emb_dim, num_heads=4))
        
        #why is inchannels here last downchannel*2?
        self.ups = nn.ModuleList([])
        for i in reversed(range(len(self.down_channels)-1)):
            self.ups.append(UpSampleBlock(self.down_channels[i]*2, self.down_channels[i-1] if i!=0 else 16, self.t_emb_dim, up_sample=self.down_sample[i], num_heads=4))
          
        #these are to be done on the final upsampled output.
        #why? -> conv => being done to get us to the same number of channels as input image. norm => to make sure input to conv isnt wild
        self.norm_out = nn.GroupNorm(8, 16)
        self.SiLU = nn.SiLU()# Define SiLU activation function as a class attribute
        self.conv_out = nn.Conv2d(16, im_channels, kernel_size=3, padding=1)
        
    def forward(self, x, t):
        out = self.conv_in(x)
        t_emb = get_time_embedding(t, self.t_emb_dim)
        t_emb = self.t_proj(t_emb)
        
        #downsampling i.e encoder part
        down_outs=[] #need to save downblock outputs in a list so that it can be given to upsampled output via skip connection
        for down in self.downs:
            down_outs.append(out)
            out=down(out, t_emb)
        
        #no need to save mid outs in a list
        for mid in self.mids:
            out = mid(out, t_emb)
        
        for up in self.ups:
            down_out=down_outs.pop()
            out = up(out, down_out, t_emb) 
            
        out = self.norm_out(out)
        out = self.SiLU(out)
        out = self.conv_out(out)
        
        return out