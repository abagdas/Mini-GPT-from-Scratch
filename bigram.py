from __future__ import annotations
import torch
import torch.nn as nn

class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size:int):
        super().__init__()
        self.embedding_table=nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx:torch.Tensor,targets:torch.Tensor|None=None)->tuple[torch.Tensor,torch.Tensor|None]:
        loss=None
        logits=self.embedding_table(idx)

        if targets is not None:
            B,T,C=logits.shape
            loss=nn.functional.cross_entropy(logits.view(B*T,C),targets.view(B*T))

        return (logits,loss)

    def generate(self,idx:torch.Tensor,max_new_tokens:int)->torch.Tensor:
        for _ in range(max_new_tokens):
            logits,_=self(idx)
            probs=nn.functional.softmax(logits[:,-1,:],dim=-1)
            new_idx=torch.multinomial(probs,num_samples=1)
            idx=torch.concat((idx,new_idx),dim=1)

        return idx
