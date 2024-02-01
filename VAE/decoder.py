import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_Decoder(nn.Sequence):
    def __init__(self,channels:int):
        super().__init__(
            nn.Conv2d(4,4,kernel_size=1,padding=0),
            nn.Conv2d(4,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            VAE_ResidualBlock(512,512),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,kernel_size=3,padding=1),
            VAE_ResidualBlock(512,256),
            VAE_ResidualBlock(256,256),
            VAE_ResidualBlock(256,256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,256,kernel_size=3,padding=1),
            VAE_ResidualBlock(256,128),
            VAE_ResidualBlock(128,128),
            VAE_ResidualBlock(128,128),
            nn.GroupNorm(32,128),
            nn.SiLU(),
            nn.Conv2d(128,3,kernel_size=3,padding=1),
            
            
        )
        
    def forward(self,x: torch.Tensor)->torch.Tensor:
        x/=0.18215
        for module in self:
            x= module(x)
        return x
            
    


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
        