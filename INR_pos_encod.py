import torch

def pose_encod(B, x):
        # postional feature mapping of input
        X_proj= 2 * torch.pi * torch.matmul(x, B.T)
        cos = torch.cos(X_proj)
        sin = torch.sin(X_proj)
        out = torch.cat([cos, sin], dim=-1)
        
        return out