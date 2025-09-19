import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoTokenizer, AutoModel
from transformers.optimization import AdamW

from sklearn.preprocessing import LabelEncoder

from model import *
from config import *
from dataloader import *


def inference(args):
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    model=ProtoClassifier()

    train_dataset=ProtoDataset(args,args.save_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            drop_last=False, pin_memory=False, 
                            num_workers=0, collate_fn=collate_fn)    
    for idx, batch in enumerate(train_loader):
        print(batch['input_ids'].shape)
        
        output=model()
        
        
        




def train(args):
    
    device='mps' if torch.backends.mps.is_available() else 0
    
    model=ProtoClassifier()
    

    train_dataset=ProtoDataset(args,tokenizer,args.save_path)
    
    val_dataset=ProtoDataset(args,tokenizer,args.test_path)
    
    test_dataset=ProtoDataset(args,tokenizer,args.val_path)
    
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            drop_last=False, pin_memory=False, 
                            num_workers=0, collate_fn=collate_fn)    
    
    
    val_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            drop_last=False, pin_memory=False, 
                            num_workers=0, collate_fn=collate_fn)   
    
    

    test_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            drop_last=False, pin_memory=False, 
                            num_workers=0, collate_fn=collate_fn)   
    
    
    
    optim=AdamW(model.parameters(),lr=3e-5,weight_decay=0.01,eps=1e-8)

    
    
    model.train()
    
    train_loss=0
    
    for epochs in range(5):
        
        for i,batch in enumerate(train_dataset):
            
            out,loss=model(batch)
            
            optim.zero_grad()
            loss[0].backward()
            optim.step()            
            
            
            train_loss+=loss.item()    
            
        
        train_loss/=len(train_dataset)
        
        
        model.eval()
        val_loss=0        
        with torch.no_grad():
            
            for i,batch in enumerate(val_dataset):

                out,loss=model(batch)
                
                optim.zero_grad()
                loss[0].backward()
                optim.step()            
                
                
                val_loss+=loss.item()    
                
        
        val_loss/=len(val_dataset)



if __name__=="__main__":
    
    
    args=model_parse_args()
    inference(args)      