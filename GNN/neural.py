import tqdm
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch import nn
from torch_geometric.loader import DataLoader
from skorch.dataset import get_len
from skorch.dataset import unpack_data
from skorch.utils import to_tensor 
from skorch import NeuralNetClassifier
from tqdm import tqdm
from process import SliceDataset, weights
from skorch.utils import to_numpy

#Subclassing the NeuralNetClassifier to make it compatible for
#graph training
class NeuralNetGraph(NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Use torch geometric dataloader as default dataloader    
        self.iterator_train = DataLoader
        self.iterator_valid = DataLoader
    
    #Modify the loss function of NeuralNetClassifier to include 
    #pT-eta weight
    def get_loss(self, y_pred, y_true, X=None, training=False):
        y_true = to_tensor(y_true, device=self.device)
        weight = X.w.to(self.device)
        
        loss_unreduced = self.criterion_(y_pred, y_true)
        loss = (loss_unreduced*weight).mean()

        return loss

    #Modify the train step to accommodate graphs and pT-eta weight
    def train_step(self, batch, **fit_params):
        self.module_.train()
        
        inputs = batch.to(self.device)
        labels = batch.y.to(self.device)
        weight = batch.w.to(self.device)
        
        self.optimizer_.zero_grad()
        out = self.module_(inputs)
        
        loss = self.get_loss(out, labels, inputs)
        
        loss.backward()
        self.optimizer_.step()
        
        return {'loss' : loss, 'y_pred' : out}
    
    #Modify the validation step to acommodate graphs and pT-eta weight
    def validation_step(self, batch, **fit_params):
        inputs = batch.to(self.device)
        labels = batch.y.to(self.device)

        with torch.no_grad():
            out = self.module_(inputs)
            loss = self.get_loss(out, labels, inputs)
        return {
            'loss': loss,
            'y_pred': out,
        }
    
    
    def evaluation_step(self, batch, training=False):
        self.check_is_fitted()
        inputs = batch.to(self.device)
        
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.module_(inputs)
    
    def run_single_epoch(self, iterator, training, prefix, 
                        step_fn, **fit_params):
        if iterator is None:
            return

        batch_count = 0
        for batch in iterator:
            self.notify("on_batch_begin", batch=batch, training=training)
            step = step_fn(batch, **fit_params)
            self.history.record_batch(prefix + "_loss", step["loss"].item())
            batch_size = (get_len(batch[0]) 
                          if isinstance(batch, (tuple, list))
                          else get_len(batch))
            self.history.record_batch(prefix + "_batch_size", batch_size)
            self.notify("on_batch_end", batch=[batch.x, batch.y], 
                        training=training, **step)
            batch_count += 1

        self.history.record(prefix + "_batch_count", batch_count)

    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        self.check_training_readiness()
        epochs = epochs if epochs is not None else self.max_epochs

        #Instead of using X, y for training features and label, use them
        #for training features and validation features
        #The label is instead extracted in train_step, validation_step, etc.
        dataset_train, dataset_valid = X, y
        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }
        iterator_train = self.get_iterator(dataset_train, training=True)
        iterator_valid = None
        if dataset_valid is not None:
            iterator_valid = self.get_iterator(dataset_valid, training=False)

        for _ in range(epochs):
            self.notify('on_epoch_begin', **on_epoch_kwargs)

            self.run_single_epoch(iterator_train, training=True, 
                                  prefix="train",
                                  step_fn=self.train_step, **fit_params)

            self.run_single_epoch(iterator_valid, training=False, 
                                  prefix="valid",
                                  step_fn=self.validation_step, **fit_params)

            self.notify("on_epoch_end", **on_epoch_kwargs)
        return self    


#Differentiable function for distance correlation (DisCo)
#The code is taken from https://github.com/gkasieczka/DisCo/blob/master/Disco.py
def distance_corr(var_1,var_2,normedweight,power=1):
    xx = var_1.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))
    yy = var_1.repeat(len(var_1),1).view(len(var_1),len(var_1))
    amat = (xx-yy).abs()

    xx = var_2.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))
    yy = var_2.repeat(len(var_2),1).view(len(var_2),len(var_2))
    bmat = (xx-yy).abs()

    amatavg = torch.mean(amat*normedweight,dim=1)
    Amat=amat-amatavg.repeat(len(var_1),1).view(len(var_1),len(var_1))\
        -amatavg.view(-1, 1).repeat(1, len(var_1)).view(len(var_1),len(var_1))\
        +torch.mean(amatavg*normedweight)

    bmatavg = torch.mean(bmat*normedweight,dim=1)
    Bmat=bmat-bmatavg.repeat(len(var_2),1).view(len(var_2),len(var_2))\
        -bmatavg.view(-1, 1).repeat(1, len(var_2)).view(len(var_2),len(var_2))\
        +torch.mean(bmatavg*normedweight)

    ABavg = torch.mean(Amat*Bmat*normedweight,dim=1)
    AAavg = torch.mean(Amat*Amat*normedweight,dim=1)
    BBavg = torch.mean(Bmat*Bmat*normedweight,dim=1)

    if(power==1):
        dCorr=(torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight)))
    elif(power==2):
        dCorr=(torch.mean(ABavg*normedweight))**2/(torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))
    else:
        dCorr=((torch.mean(ABavg*normedweight))/torch.sqrt((torch.mean(AAavg*normedweight)*torch.mean(BBavg*normedweight))))**power
    
    return dCorr

#Subclassing the NeuralNetGraph to accommodate DisCo regularisation
class NeuralNetGraphDiscorr(NeuralNetGraph):
    def __init__(self, factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Accommodate defining the lambda factor
        self.factor = factor
    
    #Modify the loss function to include DisCo loss
    def get_loss(self, y_pred, y_true, X=None, training=False):
        mass = np.asarray(SliceDataset(X, idx=2)) 
        norm = np.ones(len(mass))
        weight = X.w.to(self.device)
        
        weightt = weights.to(self.device)
        
        #Constructing the discriminant to be decorrelated from mass
        predict = F.softmax(y_pred,dim=1)/weightt
        predicted = predict[:,-1]/sum(predict[:,i] for i in range(len(y_pred[0]) - 1))
        
        #Initialising variable to calculate DisCo loss
        y_true = to_tensor(y_true, device=self.device)
        mass = to_tensor(mass, device=self.device)
        norm = to_tensor(norm, device=self.device)
        disco = 0

        #Calculate DisCo loss for each class of background and sum over them
        for i in range(len(y_pred[0]) - 1):
            with torch.no_grad():
                mask = (y_true.squeeze() == i)
            mass_bkg = mass[mask]
            norm_bkg = norm[mask]
            pred_bkg = predicted[mask]
            disco += distance_corr(mass_bkg,pred_bkg,norm_bkg,power=1)
        
        loss_unreduced = self.criterion_(y_pred, y_true)
        loss = (loss_unreduced*weight).mean()

        #Adding DisCo loss to regularise the original loss
        return loss + self.factor * disco

