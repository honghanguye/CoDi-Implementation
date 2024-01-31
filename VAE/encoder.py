import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder import VAE_ResidualBlock,VAE_AttentionBlock

class VAE_encoder(nn.Sequential):
    
    def __init__(self):
        super().__init__(
            
    # orginal structure as in the paper
        # Batch_size 128, channel 3, height, width
    nn.Conv2d(128,3,kernel_size=3,stride=1,padding=1),
    VAE_ResidualBlock(128,128),
    VAE_ResidualBlock(128,128),
    # Batch_size 128, channel 128, height/2, width/2
    nn.Conv2d(128,128,kernel_size=3,stride=2,padding=0),
    VAE_ResidualBlock(128,256),
    VAE_ResidualBlock(256,256),
    # Batch_size 128, channel 256, height/4, width/4
    nn.Conv2d(256,256,kernel_size=3,stride=2,padding=0),
    VAE_ResidualBlock(256,512),	
    VAE_ResidualBlock(512,512),
    # Batch_size 128, channel 512, height/8, width/8
    nn.Conv2d(512,512,kernel_size=3,stride=2,padding=0),
   
    VAE_ResidualBlock(512,512),
    VAE_ResidualBlock(512,512),
    VAE_ResidualBlock(512,512),
    VAE_AttentionBlock(512),
    VAE_ResidualBlock(512,512),
    # Group normalization with batch size 32, channel 512
    nn.GroupNorm(32,512),
    nn.SiLU(),
    # Bottleneck layer, batch size 512, channel 8, same height and width
    nn.Conv2d(512,8,kernel_size=3,padding=1),
    # Bottleneck layer, batch size 8, channel 8, same height and width
    nn.Conv2d(8,8,kernel_size=3,padding=0),
        )
        
    def forward(self,x: torch.Tensor,noise: torch.Tensor)->torch.Tensor:
        for module in self:
            if getattr(module,'stride',None)==(2,2):
                x= F.pad(x,(0,1,0,1))
                
            x = module(x)
            # divide the tensor into two parts to get mean and log_var
            mean,log_var = torch.chunk(x,2,dim=1)
            # constraint the variance to be in a certain range
            log_var = torch.clamp(log_var,-30,20)
            variance = torch.exp(log_var)
            standard_deviation = torch.sqrt(variance)
            # sample from the standard normal distribution
            x= mean + standard_deviation*noise
            x*=0.18215
            return x
            
    
    
    
    
    
    

    