import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
from torch import nn
from skorch.dataset import get_len
from skorch.dataset import unpack_data
from skorch.utils import to_tensor
from skorch import NeuralNetClassifier
from process import weights, scaler, cols_inv


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

#Subclassing the NeuralNetClassifier to incorporate event weight
class NeuralNetBase(NeuralNetClassifier):
    def train_step_single(self, batch, **fit_params):
        self._set_training(True)
        
        Xi, yi = unpack_data(batch)
        Xinput = to_tensor(Xi[:,:len(Xi[0])-1], device=self.device)
        weight = to_tensor(Xi[:,-1], device=self.device)
        y_pred = self.infer(Xinput, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        loss.backward()
        return {
            'loss': loss,
            'y_pred': y_pred,
        }

    def get_loss(self, y_pred, y_true, X=None, training=False):
        weight = to_tensor(X[:,-1], device=self.device)
        y_true = to_tensor(y_true, device=self.device)
        
        #Modify the loss function of NeuralNetClassifier to include 
        #pT-eta weight
        loss_unreduced = self.criterion_(y_pred, y_true)*weight
        loss_reduced = loss_unreduced.mean()
        return loss_reduced
    
    def validation_step(self, batch, **fit_params):
        self._set_training(False)
        
        Xi, yi = unpack_data(batch)
        Xinput = to_tensor(Xi[:,:len(Xi[0])-1], device=self.device)
        weight = to_tensor(Xi[:,-1], device=self.device)
        with torch.no_grad():
            y_pred = self.infer(Xinput, **fit_params)
            loss = self.get_loss(y_pred, yi, X=Xi, training=False)
        return {
            'loss': loss,
            'y_pred': y_pred,
        }
    
    def evaluation_step(self, batch, training=False):
        self.check_is_fitted()
        
        Xi, _ = unpack_data(batch)
        Xinput = to_tensor(Xi[:,:len(Xi[0])-1], device=self.device)
        weight = to_tensor(Xi[:,-1], device=self.device)
        with torch.set_grad_enabled(training):
            self._set_training(training)
            return self.infer(Xinput)

#Subclassing the NeuralNetBase to accommodate DisCo regularisation
class NeuralNetDisCo(NeuralNetBase):
    def __init__(self, factor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor = factor
    
    #Modify the loss function to include DisCo loss
    def get_loss(self, y_pred, y_true, X=None, training=False):
        self.criterion__weight = to_tensor(self.criterion__weight, device=self.device)
        mass = scaler.inverse_transform(X[:,cols_inv])[:,2] 
        norm = np.ones(len(mass))
        
        #Constructing the discriminant to be decorrelated from mass
        predict = F.softmax(y_pred,dim=1)/self.criterion__weight
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
        
        weight = to_tensor(X[:,-1], device=self.device)
        loss_unreduced = self.criterion_(y_pred, y_true)*weight
        loss_reduced = loss_unreduced.mean()
        
        #Adding DisCo loss to regularise the original loss
        return loss_reduced + self.factor * disco
                