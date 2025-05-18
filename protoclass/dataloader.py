import json  
import os

import pandas as pd
from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence

from sklearn.preprocessing import LabelEncoder

from config import *

def proto_dataset(path: Optional[str]=None, 
                raw_path: Optional[str]=None):
    
    
    if os.path.exists(path):
        data=pd.read_csv(path)
        
        if 'perspective_labels' in data.columns:
            return data
        
        labels=map_labels(data)
        
        data['perspective_labels']=labels
        
        data.to_csv(path,index=False)
        return pd.read_csv(path)

    else:
        
        data=pd.read_json(raw_path)
        
        df_llm=defaultdict(list)

        for idx,row in data.iterrows():
            
            summary_lbls=len(row['labelled_summaries'].items())

            ans=' '.join(a for a in row['answers'])
            ans=[ans]*summary_lbls
            ques=[row['question']]*summary_lbls
            
            df_llm['question'].extend(ques)
            df_llm['answers'].extend(ans)
                    
            for (keys,values) in row['labelled_summaries'].items():
                
                df_llm['summaries'].append(values)
                
                if 'information' in keys.lower():
                    df_llm['perspective'].append('information')
                    tone_attribute = "Informative, Educational"

                    defn = "Defined as knowledge about diseases, disorders, and health-related facts, providing insights into symptoms and diagnosis."
                    
                    df_llm['perspective_definition'].append(defn)
                    df_llm['tone_attribute'].append(tone_attribute)

                    
                elif 'cause' in keys.lower():
                    df_llm['perspective'].append('cause')
                    tone_attribute = "Advisory, Recommending"
                    defn = "Defined as reasons responsible for the occurrence of a particular medical condition, symptom, or disease"
                    
                    df_llm['perspective_definition'].append(defn)
                    df_llm['tone_attribute'].append(tone_attribute)



                elif 'suggestion' in keys.lower():
                    df_llm['perspective'].append('suggestion')
                    tone_attribute = "Advisory, Recommending"
                    defn = "Defined as advice or recommendations to assist users in making informed medical decisions, solving problems, or improving health issues."
                    
                    df_llm['perspective_definition'].append(defn)
                    df_llm['tone_attribute'].append(tone_attribute)


                    
                elif 'experience' in keys.lower():
                    df_llm['perspective'].append('experience')
                    tone_attribute = "Personal, Narrative"

                    defn = "Defined as individual experiences, anecdotes, or firsthand insights related to health, medical treatments, medication usage, and coping strategies"

                    df_llm['perspective_definition'].append(defn)
                    df_llm['tone_attribute'].append(tone_attribute)

                else:
                    df_llm['perspective'].append('question')
                    tone_attribute = "Seeking Understanding"

                    defn = "Defined as inquiry made for deeper understanding."
                    
                    df_llm['perspective_definition'].append(defn)
                    df_llm['tone_attribute'].append(tone_attribute)



        pd.DataFrame(dict(df_llm)).to_csv(path,index=False)

        return pd.read_csv(path)


def map_labels(data):
    le = LabelEncoder()
    return le.fit_transform(data['perspective'])
    

class ProtoDataset(Dataset):
    
    def __init__(self,args,tokenizer,path):
        
        self.path=path
        self.raw_path=args.raw_path
        
        if os.path.exists(self.path):
            if 'train' in self.path or 'val' in self.path or 'test' in self.path:
                self.data=proto_dataset(self.path)
                                        
            else:
                self.data=proto_dataset(self.raw_path)
                        
        self.tokenizer=tokenizer
        
        
    def __len__(self):
        
        return len(self.data)
    
    
    def __getitem__(self,idx):
        
        self.answer=self.data['answers'][idx]
        
        self.label=self.data['perspective_labels'][idx]
        
        encoded_inputs=self.tokenizer(self.answer,padding=True,
                                    truncation=True,return_tensors='pt')
        
        
        return {
            'input_ids':encoded_inputs['input_ids'],
            'token_type_ids':encoded_inputs['token_type_ids'],
            'attention_mask':encoded_inputs['attention_mask'],
            'labels':torch.tensor(self.label,dtype=torch.long)
        }
    

def collate_fn(batch):
    
    _input_ids=[item['input_ids'][0] for item in batch]
    
    _token_ids=[item['token_type_ids'][0] for item in batch]
    
    _attention_ids=[item['attention_mask'][0] for item in batch]
    
    input_ids=pad_sequence(_input_ids,batch_first=True).unsqueeze(1)
    
    token_ids=pad_sequence(_token_ids,batch_first=True).unsqueeze(1)

    attention_ids=pad_sequence(_attention_ids,batch_first=True).unsqueeze(1)
    
    
    return {
        'input_ids':input_ids,
        'token_type_ids':token_ids,
        'attention_mask':attention_ids,
        'labels':torch.tensor([item['labels'] for item in batch],dtype=torch.long)
}



    
if __name__=="__main__":
    
    
    args=model_parse_args()    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    train_dataset=ProtoDataset(args,tokenizer,args.save_path)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                            drop_last=False, pin_memory=False, 
                            num_workers=0, collate_fn=collate_fn)
    
    
    for i, batch in enumerate(train_loader):
        
        
        print(batch['input_ids'].shape)
        
        if i>5:break
        
        
