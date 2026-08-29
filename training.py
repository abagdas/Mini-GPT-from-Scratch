from __future__ import annotations
import torch
import torch.nn as nn
from data import get_batch

def evaluate(model:nn.Module,train_data:torch.Tensor,validation_data:torch.Tensor,block_size:int,batch_size:int,eval_iters:int)->dict[str,float]:
    train_lossi,val_lossi=0.0,0.0
    was_train=model.training
    with torch.inference_mode():
        model.eval()
        for _ in range(eval_iters):
            xtrb,ytrb=get_batch(train_data,block_size,batch_size)
            xvalb,yvalb=get_batch(validation_data,block_size,batch_size)
            _,train_loss=model(xtrb,ytrb)
            _,val_loss=model(xvalb,yvalb)
            train_lossi+=train_loss.item()/eval_iters
            val_lossi+=val_loss.item()/eval_iters

    model.train() if was_train else model.eval()
    return {"train":train_lossi,"validation":val_lossi}

def train_steps(model:nn.Module,data:torch.Tensor,optimizer:torch.optim.Optimizer,block_size:int,batch_size:int,steps:int)->list[float]:
    lossi=[]
    for _ in range(steps):
        xb,yb=get_batch(data,block_size,batch_size)
        _,loss=model(xb,yb)
        lossi.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return lossi

