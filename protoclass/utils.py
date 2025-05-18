import torch
import numpy as np

from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm

class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, score_at_min1=0,patience=100, verbose=False, delta=0, path='checkpoint.pt',
                trace_func=print,save_epochwise=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = score_at_min1
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.state_dict_list=[None]*patience
        self.improved=0
        self.stop_update=0
        self.save_model_counter=0
        self.save_epochwise=save_epochwise
        self.times_improved=0
        self.activated=False
    def activate(self,s1,s2):
        if not self.activated and s1>0 and s2>0: self.activated=True
    def __call__(self, score, epoch,model):
        if not self.activated: return None
        self.save_model_counter = (self.save_model_counter + 1) % 4
        if not self.stop_update:
            if self.verbose:
                self.trace_func(f'\033[91m The val score  of epoch {epoch} is {score:.4f} \033[0m')
            if score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'\033[93m EarlyStopping counter: {self.counter} out of {self.patience} \033[0m')
                if self.counter >= self.patience:
                    self.early_stop = True
                self.improved=0
            else:
                self.save_checkpoint(score, model,epoch)
                self.best_score = score
                self.counter = 0
                self.improved=1
        else:
            self.improved=0 #not needed though

    def save_checkpoint(self, score, model, epoch):
        '''Saves model when validation loss decrease.'''
        # if self.verbose:
        self.times_improved+=1
        self.trace_func(f'\033[92m Validation score improved ({self.best_score:.4f} --> {score:.4f}). \033[0m')
        if self.save_epochwise:
            path=self.path+"_"+str(self.times_improved)+"_"+str(epoch)
        else:
            path=self.path
        torch.save(model.state_dict(), path)

def evaluate(dl,model_new=None,path=None,modelclass=None):
    assert (model_new is not None) ^ (path is not None)
    if path is not None:
        model_new=modelclass().cuda()
        model_new.load_state_dict(torch.load(path))
    loader = tqdm(dl, total=len(dl), unit="batches")
    total_len=0
    model_new.eval()    
    with torch.no_grad():
        total_loss=0
        tts=0
        y_pred=[]
        y_true=[]
        for batch in loader:
            input_ids,attn_mask,y=batch
            classfn_out,loss=model_new(input_ids,attn_mask,y,use_decoder=False,use_classfn=1)
#             print(classfn_out.detach().cpu())
            if classfn_out.ndim==1:
                predict=torch.zeros_like(y)
                predict[classfn_out>0]=1
            else:
                predict=torch.argmax(classfn_out,dim=1)
                
            y_pred.append(predict.cpu().numpy())
            y_true.append(y.cpu().numpy())
            total_loss+=(len(input_ids)*loss[0].item())
            total_len+=len(input_ids)
#             torch.cuda.empty_cache()            
        total_loss=total_loss/total_len
        mac_prec,mac_recall,mac_f1_score,_=precision_recall_fscore_support(np.concatenate(y_true),np.concatenate(y_pred),labels=[0,1])
#         mic_prec,mic_recall,mic_f1_score,_=precision_recall_fscore_support(np.concatenate(y_true),np.concatenate(y_pred),labels=[0,1])
        mic_prec,mic_recall,mic_f1_score,_=0,0,0,0

    return total_loss,mac_prec,mac_recall,mac_f1_score,mic_prec,mic_recall,mic_f1_score
