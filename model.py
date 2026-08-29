from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import CausalSelfAttentionHead,MultiHeadAttention

class SingleHeadAttentionLeanguageModel(nn.Module):
    def __init__(self,block_size:int,n_embd:int,vocab_size:int,dropout:float):
        super().__init__()
        self.token_embedding_table=nn.Embedding(vocab_size,n_embd)
        self.position_embedding_table=nn.Embedding(block_size,n_embd)
        self.lm_head=nn.Linear(n_embd,vocab_size)
        self.atttention=CausalSelfAttentionHead(n_embd,n_embd,block_size,dropout)

    def embed_tokens_and_positions(self,idx:torch.Tensor)->torch.Tensor:
        _,T=idx.shape
        token_embed=self.token_embedding_table(idx)
        pos_ids=torch.arange(T,device=idx.device)
        position_embed=self.position_embedding_table(pos_ids)

        return token_embed+position_embed

    def forward(self,x:torch.Tensor,targets:torch.Tensor|None=None)->tuple[torch.Tensor,torch.Tensor|None]:
        loss=None
        x=self.embed_tokens_and_positions(x)
        x=self.atttention(x)
        logits=self.lm_head(x)

        if targets is not None:
            B,T,V=logits.shape
            loss=F.cross_entropy(logits.view(B*T,V),targets.view(B*T))

        return (logits,loss)

class FeedForward(nn.Module):
    def __init__(self,dropout:float,input_dim:int):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,input_dim*4),nn.ReLU(),nn.Linear(input_dim*4,input_dim),nn.Dropout(dropout))

    def forward(self,x:torch.Tensor)->torch.Tensor:
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self,dropout:float,input_dim:int,num_heads:int,block_size:int):
        super().__init__()
        self.ln1=nn.LayerNorm(input_dim)
        self.ln2=nn.LayerNorm(input_dim)
        self.ffwd=FeedForward(dropout,input_dim)
        head_size=input_dim//num_heads
        self.sa=MultiHeadAttention(input_dim,num_heads,block_size,head_size,dropout)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        x=x+self.sa(self.ln1(x))
        out=x+self.ffwd(self.ln2(x))

        return out

class StackedTransformerLanguageModel(nn.Module):
    def __init__(self,vocab_size:int,block_size:int,num_layers:int,dropout:float,input_dim:int,num_heads:int):
        super().__init__()
        self.block_size=block_size
        self.token_embedding_table=nn.Embedding(vocab_size,input_dim)
        self.position_embedding_table=nn.Embedding(block_size,input_dim)
        self.lm_head=nn.Linear(input_dim,vocab_size)
        self.ln_f=nn.LayerNorm(input_dim)
        self.blocks=nn.Sequential(*[TransformerBlock(dropout,input_dim,num_heads,block_size) for _ in range(num_layers)])

    def forward(self,idx:torch.Tensor,targets:torch.Tensor|None=None)->tuple[torch.Tensor,torch.Tensor|None]:
        _,T=idx.shape
        loss=None
        token_embed=self.token_embedding_table(idx)
        pos_ids=torch.arange(T,device=idx.device)
        pos_embed=self.position_embedding_table(pos_ids)
        x=token_embed+pos_embed
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)

        if targets is not None:
            B,T,V=logits.shape
            loss=F.cross_entropy(logits.view(B*T,V),targets.view(B*T))

        return (logits,loss)

    def generate(self,idx:torch.Tensor,max_new_tokens:int)->torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-self.block_size:]
            logits,_=self(idx_cond)
            last_logits=logits[:,-1,:]
            probs=F.softmax(last_logits,-1)
            new_idx=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,new_idx),dim=1)

        return idx
