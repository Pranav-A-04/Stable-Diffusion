import torch
import torch.nn as nn

# Resnet block -> 2 conv layers with skip connection and time embedding (only in diffusion model)
# time embedding is optional because autoencoder doesnt use it
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_channels, t_emb_dim=None):
        super().__init__()
        self.resnet_conv_first=nn.Sequential(
            nn.GroupNorm(norm_channels, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        
        self.t_emb_layers=nn.Sequential(
            nn.SiLU(),
            nn.Linear(t_emb_dim, out_channels)
        ) if t_emb_dim is not None else nn.Identity()
        
        self.resnet_conv_second=nn.Sequential(
            nn.GroupNorm(norm_channels, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.residual_input_conv=nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x, t_emb):
        #first resnet block
        resnet_input=x
        out = self.resnet_conv_first(x)
        if t_emb is not None:
            out = out + self.t_emb_layers(t_emb)[:, :, None, None]
        out = self.resnet_conv_second(out)
        out = out + self.residual_input_conv(resnet_input)
        
        return out

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels, num_heads, norm_channels):
        super().__init__()
        self.norm=nn.GroupNorm(norm_channels, in_channels)
        self.attention=nn.MultiheadAttention(in_channels, num_heads, batch_first=True)
        self.residual_input_conv=nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
    def forward(self, x):
        #x is of shape (batch_size, channels, height, width)
        resnet_input=x
        batch_size, channels, height, width = x.shape
        out = self.norm(x)
        out = out.view(batch_size, channels, height * width).permute(0, 2, 1)  # (batch_size, height*width, channels)
        out, _ = self.attention(out, out, out)  # Self-attention
        out = out.permute(0, 2, 1).view(batch_size, channels, height, width)  # Back to (batch_size, channels, height, width)
        out = out + self.residual_input_conv(resnet_input)
        return out

#downblock
class DownSampleBlock(nn.Module):
  def __init__(self, in_channels, out_channels, down_sample, num_heads, norm_channels, t_emb_dim=None, self_attention=False):
    super().__init__()
    self.down_sample=down_sample
    self.resnet_block = ResnetBlock(in_channels, out_channels, norm_channels, t_emb_dim)
    self.self_attn_block = SelfAttentionBlock(out_channels, num_heads, norm_channels) if self_attention else nn.Identity()
    self.down_sample_conv=nn.Conv2d(out_channels, out_channels, kernel_size=4,
                                    stride=2, padding=1) if self.down_sample else nn.Identity()

  def forward(self, x, t_emb):
    out = x
    #RESNET BLOCK
    out=self.resnet_block(out, t_emb)

    #ATTENTION BLOCK
    out=self.self_attn_block(out)

    #downsample
    out=self.down_sample_conv(out)
    return out

class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, t_emb_dim, num_heads, norm_channels):
        super().__init__()
        self.first_resnet_block = ResnetBlock(in_channels, out_channels, t_emb_dim, norm_channels=norm_channels)
        self.self_attention_block = SelfAttentionBlock(out_channels, num_heads, norm_channels)
        self.second_resnet_block = ResnetBlock(out_channels, out_channels, t_emb_dim, norm_channels=norm_channels)
    

    def forward(self, x, t_emb):
        out = x
        #first resnet block
        out = self.first_resnet_block(out, t_emb)
        
        #the entire part below i.e attention block + second resnet block is taken together as a single block which can have multiple layers i.e the below part can be in a for loop
        #attention block
        out = self.self_attention_block(out)
        
        #second resnet block
        out = self.second_resnet_block(out, t_emb)
        
        return out
    
class UpSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample, num_heads, norm_channels, t_emb_dim=None, self_attention=False):
        super().__init__()
        self.up_sample=up_sample
        self.resnet_block = ResnetBlock(in_channels, out_channels, t_emb_dim, norm_channels=norm_channels)
        self.self_attention_block = SelfAttentionBlock(out_channels, num_heads, norm_channels) if self_attention else nn.Identity()
        
        #upsampling is done via transposed convolution
        self.up_sample_conv=nn.ConvTranspose2d(in_channels//2, in_channels//2, kernel_size=4,
                                               stride=2, padding=1) if self.up_sample else nn.Identity()
        
    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1) #skip connection
        out = x
    
        #resnet block
        out = self.resnet_block(out, t_emb)
        
        #attention block
        out = self.self_attention_block(out)
        
        return out