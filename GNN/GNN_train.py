import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader
from torch import nn
from tqdm import tqdm

from process import weights, train_dataset
from module import ParticleNet, ParticleNetMini
from neural import NeuralNetGraph, NeuralNetGraphDiscorr
from scoring import aucval, accept, reject, signif
from scoring import accept_epoch, reject_epoch
from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint
from sklearn.pipeline import Pipeline

torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Constructing callbacks for monitoring the training 
reje_e = EpochScoring(reject_epoch, lower_is_better=False)
acce_e = EpochScoring(accept_epoch, lower_is_better=False)
stop = EarlyStopping(monitor='valid_acc', patience=3, lower_is_better=False, 
                      threshold=0.0001, load_best=True)
cp = Checkpoint(monitor='valid_acc_best')


#Dividing the train dataset into train and validation (for monitoring)
torch.manual_seed(1234)
train_dataset = train_dataset.shuffle()

a = int(len(train_dataset))
b = int(a*0.8)
X_train = train_dataset[:b]
X_valid = train_dataset[b:a]

#Example of training model for certain configuration without DisCo
model_R020 = NeuralNetGraph(
    module = ParticleNetMini([64, 128, 256, 512], 128, 0.7578, 16, 10, num_classes = len(weights)),
    criterion = nn.CrossEntropyLoss,
    optimizer = optim.Adam,
    criterion__reduction = 'none',
    criterion__weight = weights,
    verbose = 10,
    optimizer__lr = 0.04636,
    batch_size= 256,
    classes = [0,1,2,3,4],
    train_split=None,
    device = device,
    max_epochs = 11,
    iterator_train__num_workers = 16,
    iterator_valid__num_workers = 16,
    iterator_train__pin_memory = False,
    iterator_valid__pin_memory = False,
    callbacks = [reje_e, acce_e, stop, cp]
)
#Training
model_R020.fit(X_train, X_valid)
#Saving the model for evaluation use
model_R020.save_params(f_params='model_29th/model_R020.pkl')

#Example of training model for certain configuration with DisCo
model_M020 = NeuralNetGraphDiscorr(
    factor = 3,
    module = ParticleNetMini([32, 64, 128], 128, 0.3242, 12, 10, num_classes = len(weights)),
    criterion = nn.CrossEntropyLoss,
    optimizer = optim.Adam,
    criterion__reduction = 'none',
    criterion__weight = weights,
    verbose = 10,
    optimizer__lr = 0.0002,
    batch_size= 1024,
    classes = [0,1,2,3,4],
    train_split=None,
    device = device,
    max_epochs = 11,
    iterator_train__num_workers = 16,
    iterator_valid__num_workers = 16,
    iterator_train__pin_memory = False,
    iterator_valid__pin_memory = False,
    callbacks = [reje_e, acce_e, stop, cp]
)
#Training
model_M020.fit(X_train, X_valid)
#Saving the model for evaluation use
model_M020.save_params(f_params='model_29th/model_M020.pkl')


