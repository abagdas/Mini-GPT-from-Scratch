from __future__ import annotations
import torch

def build_vocabulary(text:str)->tuple[dict[str,int],dict[int,str]]:
    chars=sorted(set(text))
    stoi={s:i for i,s in enumerate(chars)}
    itos={i:s for i,s in enumerate(chars)}

    return (stoi,itos)

def encode(text:str,stoi:dict[str,int])->list[int]:
    out=[stoi[s] for s in text]

    return out

def decode(token_ids:list[int],itos:dict[int,str])->str:
    out="".join([itos[i] for i in token_ids])

    return out

def text_to_tensor(text:str,stoi:dict[str,int])->torch.Tensor:
    out=torch.tensor(encode(text,stoi),dtype=torch.long)

    return out

def split_train_validation(data:torch.Tensor,train_fraction:float)->tuple[torch.Tensor,torch.Tensor]:
    n=int(train_fraction*len(data))
    train_data=data[:n]
    validation_data=data[n:]

    return (train_data,validation_data)

def get_batch(data:torch.Tensor,block_size:int,batch_size:int)->torch.Tensor:
    idx=torch.randint(0,len(data)-block_size,(batch_size,))
    xb,yb=[],[]
    for i in idx:
        xb.append(data[i:i+block_size])
        yb.append(data[i+1:i+block_size+1])

    return (torch.stack(xb),torch.stack(yb))