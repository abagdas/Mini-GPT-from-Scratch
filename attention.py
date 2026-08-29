from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

def causal_mean_loop(x:torch.Tensor)->torch.Tensor:
    B,T,C=x.shape
    mean=torch.zeros_like(x)
    for b in range(B):
        for t in range(T):
            mean[b,t]=torch.mean(x[b,:t+1,:],dim=0)

    return mean

def causal_mean_weights(sequence_length:int,device:torch.device|None,dtype:torch.dtype=torch.long)->torch.Tensor:
    ones=torch.ones(sequence_length,sequence_length,dtype=dtype,device=device)
    lower_triangle=torch.tril(ones)
    out=lower_triangle/torch.sum(lower_triangle,dim=1,keepdim=True)

    return out

def causal_softmax_weights(sequence_length:int,device:torch.device|None,dtype:torch.dtype)->torch.Tensor:
    ones=torch.ones(sequence_length,sequence_length,dtype=dtype,device=device)
    lower_triangle=torch.tril(ones)
    mask=torch.masked_fill(lower_triangle,lower_triangle==0,float("-inf"))
    out=F.softmax(mask,dim=-1)

    return out

def causal_mean_matmul(x:torch.Tensor)->torch.Tensor:
    _,T,_=x.shape
    wei=causal_mean_weights(T,x.device,x.dtype)
    out=wei@x

    return out

def scaled_attention_scores(query:torch.Tensor,key:torch.Tensor)->torch.Tensor:
    H=query.shape[-1]
    out=query@key.transpose(-2,-1)*H**-0.5 #normalize

    return out

def scaled_dot_attention_scores(querry:torch.Tensor,key:torch.Tensor,value:torch.Tensor)->tuple[torch.Tensor,torch.Tensor]:
    scores=scaled_attention_scores(querry,key)
    weights=F.softmax(scores,dim=-1)
    out=weights@value

    return (weights,out)

class CausalSelfAttentionHead(nn.Module):
    def __init__(self,input_dim:int,head_size:int,block_size:int,dropout:float):
        super().__init__()
        self.register_buffer("tril",torch.tril(torch.ones(block_size,block_size)))
        self.key=nn.Linear(input_dim,head_size,bias=False)
        self.query=nn.Linear(input_dim,head_size,bias=False)
        self.value=nn.Linear(input_dim,head_size,bias=False)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        _,T,_=x.shape
        k=self.key(x)
        q=self.query(x)
        v=self.value(x)
        scores=q@k.transpose(-2,-1)*k.shape[-1]**-0.5 #normalize
        scores=torch.masked_fill(scores,self.tril[:T,:T]==0,float("-inf"))
        weights=F.softmax(scores,dim=-1)
        weights=self.dropout(weights)
        out=weights@v

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,input_dim:int,num_heads:int,block_size:int,head_size:int,dropout:float):
        super().__init__()
        self.heads=nn.ModuleList([CausalSelfAttentionHead(input_dim=input_dim,head_size=head_size,block_size=block_size,dropout=dropout) for _ in range(num_heads)])
        self.proj=nn.Linear(num_heads*head_size,input_dim)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=torch.cat([head(x) for head in self.heads],dim=-1)
        x=self.proj(x)
        out=self.dropout(x)

        return out