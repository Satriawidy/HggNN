import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from tqdm import tqdm

from process import weights, X_train_tensor, y_train_tensor
from module import DNN, DNNMini
from neural import NeuralNetBase, NeuralNetDisCo
from scoring import aucval, accept, reject, signif
from scoring import auc_epoch, accept_epoch, reject_epoch, signif_epoch
from scoring import cutval_epoch, mutual_epoch, JSDinv_epoch

from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Constructing callbacks for monitoring the training 
reje_e = EpochScoring(reject_epoch, lower_is_better=False)
acce_e = EpochScoring(accept_epoch, lower_is_better=False)
stop = EarlyStopping(monitor='valid_acc', patience=4, lower_is_better=False, 
                      threshold=0.001, load_best=True)
cp = Checkpoint(monitor='valid_acc_best')

#Example of training model for certain configuration with DisCo
model_0006 = NeuralNetDisCo(
                     factor = 2.5,
                     module = DNNMini([64,64,64,64],0.73, 15, len(weights)),
                     criterion = nn.CrossEntropyLoss,
                     criterion__reduction = 'none',
                     criterion__weight = weights,
                     optimizer = optim.Adam,
                     optimizer__lr = 0.003,
                     batch_size = 4096,
                     max_epochs = 15,
                     verbose = 10,
                     device = device,
                     callbacks = [stop, cp]
                     )
#Training
model_0006.fit(X_train_tensor, y_train_tensor)
#Saving the model for evaluation use
model_0006.save_params(f_params='model_29/model_0006.pkl')

model_0007 = NeuralNetDisCo(
                     factor = 3.0,
                     module = DNNMini([64,64,64,64],0.73, 15, len(weights)),
                     criterion = nn.CrossEntropyLoss,
                     criterion__reduction = 'none',
                     criterion__weight = weights,
                     optimizer = optim.Adam,
                     optimizer__lr = 0.003,
                     batch_size = 4096,
                     max_epochs = 15,
                     verbose = 10,
                     device = device,
                     callbacks = [stop, cp]
                     )

model_0007.fit(X_train_tensor, y_train_tensor)
model_0007.save_params(f_params='model_29/model_0007.pkl')