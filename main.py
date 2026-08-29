from __future__ import annotations
from data import build_vocabulary,text_to_tensor,split_train_validation,encode,decode
import torch
from bigram import BigramLanguageModel
from training import evaluate,train_steps
from model import StackedTransformerLanguageModel
from pathlib import Path

BATCH_SIZE=16
BLOCK_SIZE=32
INPUT_DIM=32
NUM_HEADS=4
NUM_LAYERS=2
DROPOUT=0.1
DEVICE="cuda" if torch.cuda.is_available() else "cpu"
TRAIN_FRACTION=0.9
LR=1e-3
EVAL_ITERS=100
STEPS=100
MAX_NEW_TOKENS=200

text_path=Path(__file__).with_name("input.txt")
text=text_path.read_text(encoding="utf-8")

stoi,itos=build_vocabulary(text)
data=text_to_tensor(text,stoi).to(DEVICE)
train_data,val_data=split_train_validation(data,TRAIN_FRACTION)

bigram_model=BigramLanguageModel(len(stoi)).to(DEVICE)
bigram_optimizer=torch.optim.AdamW(bigram_model.parameters(),LR)
bigram_before=evaluate(bigram_model,train_data,val_data,BLOCK_SIZE,BATCH_SIZE,EVAL_ITERS)
bigram_train_history=train_steps(bigram_model,train_data,bigram_optimizer,BLOCK_SIZE,BATCH_SIZE,STEPS)
bigram_after=evaluate(bigram_model,train_data,val_data,BLOCK_SIZE,BATCH_SIZE,EVAL_ITERS)

transformer_model=StackedTransformerLanguageModel(len(stoi),BLOCK_SIZE,NUM_LAYERS,DROPOUT,INPUT_DIM,NUM_HEADS).to(DEVICE)
transformer_optimizer=torch.optim.AdamW(transformer_model.parameters(),LR)
transformer_before=evaluate(transformer_model,train_data,val_data,BLOCK_SIZE,BATCH_SIZE,EVAL_ITERS)
transformer_train_history=train_steps(transformer_model,train_data,transformer_optimizer,BLOCK_SIZE,BATCH_SIZE,STEPS)
transformer_after=evaluate(transformer_model,train_data,val_data,BLOCK_SIZE,BATCH_SIZE,EVAL_ITERS)

prompt="\n"
context=torch.tensor([encode(prompt,stoi)],dtype=torch.long,device=DEVICE)
bigram_model.eval()
transformer_model.eval()
with torch.inference_mode():
    bigram_ids=bigram_model.generate(context,MAX_NEW_TOKENS)
    transformer_ids=transformer_model.generate(context,MAX_NEW_TOKENS)

print(f"bigram loss before after\t{bigram_before}----------->{bigram_after}\n")
print(decode(bigram_ids[0].tolist(),itos))
print("-"*50)
print(f"transformer loss before after\t{transformer_before}--------------->{transformer_after}\n")
print(decode(transformer_ids[0].tolist(),itos))