import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim

from torch import nn
from tqdm import tqdm

from process import weights, X_train_tensor, y_train_tensor
from module import DNN, DNNMini
from neural import NeuralNetBase, NeuralNetDisCo
from scoring import aucval, accept, reject
from scoring import accept_epoch, reject_epoch
from evaluation import model_eval

from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint

torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reje_e = EpochScoring(reject_epoch, lower_is_better=False)
acce_e = EpochScoring(accept_epoch, lower_is_better=False)
stop = EarlyStopping(monitor='valid_acc', patience=3, lower_is_better=False, 
                      threshold=-0.001, load_best=True)
cp = Checkpoint(monitor='valid_acc_best')

model_0001 = NeuralNetBase(
                     module = DNNMini([64,64,64,64],0.73, 15, len(weights)),
                     criterion = nn.CrossEntropyLoss,
                     criterion__reduction = 'none',
                     criterion__weight = weights,
                     optimizer = optim.Adam,
                     optimizer__lr = 0.003,
                     batch_size = 512,
                     max_epochs = 15,
                     verbose = 10,
                     device = device,
                     callbacks = [stop, cp]
                     )

model_0001.initialize()  # This is important!
model_0001.load_params(f_params='model_29/model_0001.pkl')

model_0002 = NeuralNetDisCo(
                     factor = 0.5,
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

model_0002.initialize()  # This is important!
model_0002.load_params(f_params='model_29/model_0002.pkl')

model_0003 = NeuralNetDisCo(
                     factor = 1.0,
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

model_0003.initialize()  # This is important!
model_0003.load_params(f_params='model_29/model_0003.pkl')

model_0004 = NeuralNetDisCo(
                     factor = 1.5,
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

model_0004.initialize()  # This is important!
model_0004.load_params(f_params='model_29/model_0004.pkl')

model_0005 = NeuralNetDisCo(
                     factor = 2.0,
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

model_0005.initialize()  # This is important!
model_0005.load_params(f_params='model_29/model_0005.pkl')

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

model_0006.initialize()  # This is important!
model_0006.load_params(f_params='model_29/model_0006.pkl')

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

model_0007.initialize()  # This is important!
model_0007.load_params(f_params='model_29/model_0007.pkl')

models = [model_0001, model_0007]
model_name = ['0001', '0007']
model_legend = ['DNN', r'DNN + DisCo, $\lambda=3.0$']

model_eval(models, model_name, model_legend, 2)