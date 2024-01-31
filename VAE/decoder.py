import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):
    
    def __init__(self,channels:int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32,channels)
        self.attention = SelfAttention(channels)
        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        
        residue = x
        
        n,c,h,w = x.shape
        # change the shape of the tensor to help attending to each pixel
        x = x.view(n,c,h*w)
        x= x.transpose(-1,-2)
        x = self.attention(x)
        x= x.transpose(-1,-2)
        x = x.view(n,c,h,w)
        x = self.group_norm(x)
        x += residue
        return x
        
        
        
        
        
    

 



class VAE_ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(32,in_channels)
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.norm2 = nn.GroupNorm(32,out_channels)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        
        self.activation = nn.SiLU()
        
        if in_channels== out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
    
    
    def forward(self,x: torch.Tensor)->torch.Tensor:
        residual = self.residual_layer(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        return x+residual
        