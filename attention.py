import torch
import torch.nn as nn
import torch.nn.functional as F
import math


        

class SelfAttention(nn.Module):
    
    def __init__(self,n_heads:int,embed_dim:int,in_proj_bias=True,out_proj_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        # define the big matrix W
        self.in_proj = nn.Linear(embed_dim,3*embed_dim,bias=in_proj_bias)
        self.out_proj = nn.Linear(embed_dim,embed_dim,bias=out_proj_bias)
        
    def forward(self, x: torch.Tensor,causal_mask=False)-> torch.Tensor:
        input_shape = x.shape
        batch_size, sequence_lenghth, embed_dim = input_shape
        intermediate_shape = (batch_size,sequence_lenghth,self.n_heads,self.head_dim)
        # divide W into 3 tensors with the size of the original tensor for query, key and value
        q,k,v = torch.chunk(self.in_proj,3,dim=-1)
        # define each tensor's shape
        q = q.view(*intermediate_shape).transpose(1,2)
        k = k.view(*intermediate_shape).transpose(1,2)
        v = v.view(*intermediate_shape).transpose(1,2)
        # calculate the attention score
        weight = q@k.transpose(-1,-2)
        
        if causal_mask:
            # create a mask to prevent the model from attend to the future
            mask = torch.tril(torch.ones_like(weight),diagonal=0)
            weight = weight.masked_fill(mask==0,float('-inf'))
        
        weight = weight / math.sqrt(self.head_dim)
        weight = F.softmax(weight,dim=-1)
        
        # caclulate the output
        output = weight@v
        output = output.transpose(1,2).reshape(batch_size,sequence_lenghth,embed_dim)
        output = self.out_proj(output)
        return output
        
        
        
        
        