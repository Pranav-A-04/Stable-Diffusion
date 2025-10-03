import torch
import torch.nn as nn

#sinusoidal position embedding(section 3.3)
#basically a positional encoding to give each timestep a distinct representation.
def get_time_embedding(time_steps, t_emb_dim):
    factor=10000**((torch.arange(
        start=0, end=t_emb_dim//2, device=time_steps.device)/(t_emb_dim//2)))
    t_emb=time_steps[:, None].repeat(1, t_emb_dim//2)/factor
    t_emb=torch.cat([torch.sin(t_emb), torch.cos(t_emb)], dim=-1)
    return t_emb