import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.loader import DataLoader
from torch import nn
from tqdm import tqdm

from process import weights
from module import ParticleNet, ParticleNetMini
from neural import NeuralNetGraph, NeuralNetGraphDiscorr
from scoring import aucval, accept, reject
from scoring import accept_epoch, reject_epoch
from evaluation import model_eval

from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint

torch.multiprocessing.set_sharing_strategy('file_system')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

reje_e = EpochScoring(reject_epoch, lower_is_better=False)
acce_e = EpochScoring(accept_epoch, lower_is_better=False)
stop = EarlyStopping(monitor='auc_epoch', patience=3, lower_is_better=False, 
                      threshold=-0.001, load_best=True)
cp = Checkpoint(monitor='auc_epoch_best')


#Initialising the saved model
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

model_R020.initialize()  # This is important!
model_R020.load_params(f_params='model_29th/model_R020.pkl')

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

model_M020.initialize()  # This is important!
model_M020.load_params(f_params='model_29th/model_M020.pkl')


models = [model_R020, model_M020]
model_name = ['R020', 'M020']
model_legend = ['GNN 20', 'GNN DisCo 20']

#Extracting discriminant and score using model_eval
#Discriminant and score is evaluated on test dataset
model_eval(models, model_name, model_legend, 'r')
