import pandas as pd
import numpy as np


import torch
import torch.nn as nn 

import transformers
from transformers import AutoTokenizer, AutoModel

from config import *


class ProtoClassifier(nn.Module):
    def __init__(self,args):
        super().__init__()

        self.args=args
        self.df=pd.read_csv(self.args.path)
        self.n_classes=self.args.n_classes
        self.n_protos=self.args.n_protos
        
        self.model= AutoModel.from_pretrained(self.args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
    
        self.d_size=self.model.config.hidden_size
                
        self.loss_fn=nn.CrossEntropyLoss(reduction='mean')    
        
        self.linear_model=nn.Sequential(
            nn.Linear(self.n_protos,self.n_protos*16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(self.n_protos*16,self.n_protos)
            
        )
    
        
        if self.args.init=="random":
            self.protos=nn.Parameter(torch.rand(self.n_protos,self.d_size))
        else:
            self.protos=self.init_proto(self.args.save_path,self.tokenizer,self.model)
            
    @staticmethod        
    def mahalnabolis_dist(input,proto):
        
        delta=input-proto
        cov=torch.cov(proto)
        mah_dist=torch.matmul(delta.T,torch.matmul(torch.linalg.inv(cov),delta))
        return mah_dist
        
    def intialize_weight(self,params,method):
        if not method:
            for params in self.parameters():
                torch.nn.init.xavier_uniform_(params)

    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)        


    def init_proto(self,path:str,tokenizer:transformers.PretrainedTokenizer,
                model:transformers.PretrainedModel):
        np.random.seed(42)

        lbls=self.df['perspective'].unique()
        
        protos=torch.empty(len(lbls),model.config.hidden_size)

        for i,label in enumerate(lbls):    
            ids=self.df.loc[self.df['perspective']==label,:].index.to_list()
            
            id_list=np.random.choice(ids,5,replace=False)
            
            answers=self.df.loc[id_list,'answers'].to_list()
            
            encoded_input=self.tokenizer(answers,padding=True,
                                    truncation=True,return_tensors='pt')  
            
            output=self.model(**encoded_input)
            output=self.mean_pooling(output,encoded_input['attention_mask'])

            output=torch.mean(output,0)
            
            protos[i]=output
            
        
        return torch.nn.Parameter(protos)        

    def forward(self,tkn_output,target):
        
        input_ids,token_ids,attn_ids=tkn_output
        output=self.model(input_ids,token_ids,attn_ids)
        
        output=self.mean_pooling(output,attn_ids)
        
        dist=torch.cdist(self.protos,output)
        
        lin_output=self.linear_model(dist)
        
        classfn_loss=self.loss_fn(lin_output,target)
    
        total_loss=classfn_loss+self.mahalnabolis_dist(output,self.protos)
        return lin_output, total_loss
        


if __name__=="__main__":
    pass
    