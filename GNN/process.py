import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.transforms import KNNGraph
from torch_geometric.data import Dataset, Data
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_cluster import knn_graph

from collections.abc import Sequence
from skorch.utils import to_numpy 


hfivesdir = r"/storage/teaching/SummerProjects/2024/s2601026/input/new_Input_Tagging_29thJuly2024.h5"
graphsdir = "/localstorage/s2601026/GNN_29"

#Subclass for processing the data
#It is initially used to convert the h5 dataset into graphs,
#and then used to access the graphs for training/evaluation
class MyOwnDataset(Dataset):
    def __init__(self, root, transform = None, pre_transform = None,
                pre_filter = None):
        super().__init__(root, transform, pre_transform, pre_filter)
    
    @property
    def raw_file_names(self):
        return 'new_Input_Tagging_v7.h5'
    
    @property
    def processed_file_names(self):
        return 'new_Input_Tagging_v7.pt'
    
    def download(self):
        pass

    #Uncomment the commented lines for converting h5 dataset into graphs
    #Comment it again to save times when only accessing the graphs
    def process(self):
        with h5py.File(hfivesdir) as f:
            #self.data = pd.DataFrame(f['LargeRJet']['2d'])
            self.ones = pd.DataFrame(f['LargeRJet']['1d'][:])

        pass
        #self.ones.replace({'flavour':{0:0, 6:1, 10:2, 23:3, 24:3, 42:4}},
        #                  inplace=True)

        #i = 0
        #for row in self.data.itertuples():
        #    #Obtaining the features of each graph node/constituent
        #    node_feats = self._get_node_features(row, i)
        
        #    #Obtaining the truth label and other various graph level
        #    #information
        #    label, mass, pT, eta, weight = self._get_labels(i)

        #    #Constructing the graph dataset
        #    data = Data(x = node_feats,
        #                m = mass,
        #                y = label,
        #                p = pT,
        #                e = eta,
        #                w = weight)
        
        #    #Saving the graph dataset
        #    torch.save(data, osp.join(self.processed_dir, f'new_Input_Tagging_{i}.pt'))
        
        #    #Lines for printing number of iterations, only used to monitor
        #    #the graph preprocessing
        #    if (i%35000==1):
        #        print(i)
        #    i += 1

    
    #Function to obtain the features of each graph node
    def _get_node_features(self, row, i):
        all_node_feats = []
        
        #Each column contain 3 calo-level information,
        #and 5 track-level information
        for column in row[1:]:
            #column_1 for track-level information
            column_1 = np.asarray([column[3],column[4],column[5],column[6],column[7]])
            #column_2 for calo-level information
            column_2 = np.asarray([column[0],column[1],column[2]])
            
            #Proceed to process more features if the node does not contain
            #NaN value
            if np.isnan(column_1[0]) == 0:
                #Add label one for track-level information
                column_3 = [1]
                column_3.extend(column_1)
                
                #Process more features
                nodes = self._preprocess(column_3, i)
                #Only append to graph node if satisfies the no-NaN criterie
                all_node_feats.append(nodes)
            
            if np.isnan(column_2[0]) == 0:
                #Add zero padding for calo-level information
                column_4 = [0, 0, 0]
                column_4.extend(column_2)
                
                #Process more features
                nodes = self._preprocess(column_4, i)
                #Only append to graph node if satisfies the no-NaN criterie
                all_node_feats.append(nodes)
                
        #Convert graph nodes into array
        all_node_feats = np.asarray(all_node_feats)
        
        return torch.tensor(all_node_feats, dtype=torch.float)
    
    #Function to process the raw track/calo-level information to
    #final graph features input
    def _preprocess(self, column, i):
        E_const = column[5] * np.cosh(column[3])
        #Graph level information for original ParticeNet input
        p_jet = self.ones['pT'][i] * np.cosh(self.ones['eta'][i])
        m_jet = self.ones['mass'][i]
        E_jet = np.sqrt(p_jet**2 + m_jet**2)
        E_pTT = np.log(column[5] / self.ones['pT'][i])
        
        #Constructing original ParticleNet input
        column[3] = column[3] - self.ones['eta'][i]
        column[4] = column[4] - self.ones['phi'][i]
        column[5] = np.log(column[5])

        R_const = np.sqrt(column[3]**2 + column[4]**2)
        
        #Adding ParticleNet original input and binned value
        #(the binned values are optional and not used further in
        # the project) 
        add = [E_pTT, np.log(E_const), np.log(E_const / E_jet), R_const, self.ones['Eta_Bin'][i], self.ones['pT_Bin'][i]]

        nodes = np.append(column, add)
        
        return nodes

    #Function to obtain truth label and other graph level information
    def _get_labels(self, i):
        label = np.asarray(self.ones['flavour'][i])
        mass = np.asarray(self.ones['mass'][i])
        pT = np.asarray(self.ones['pT'][i])
        eta = np.asarray(self.ones['eta'][i])
        weight = np.asarray(self.ones['weight'][i])

        return torch.tensor(label, dtype=torch.int64), torch.tensor(mass, dtype=torch.float), torch.tensor(pT, dtype=torch.float), torch.tensor(eta, dtype=torch.float), torch.tensor(weight, dtype=torch.float)
    
    def len(self):
        return self.ones.shape[0]

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'new_Input_Tagging_{idx}.pt'))
        return data

with h5py.File(hfivesdir) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

#Constructing the class-weight
df.replace({'flavour':{0:0, 6:1, 10:2, 23:3, 24:3, 42:4}}, inplace=True)    
value_counts = df['flavour'].value_counts()
total_count = len(df)
weights = total_count / value_counts.sort_index().values
weights = torch.tensor(weights, dtype=torch.float)

#Call for processing/accessing dataset, the graphsdir is where the dataset
#saved
dataset = MyOwnDataset(root=graphsdir)

#For shuffling the graphs and dividing into training and testing dataset
#The seed is fixed for now
torch.manual_seed(1234)
dataset = dataset.shuffle()
a = int(len(dataset))
b = int(a*0.9)
train_dataset = dataset[:b]
test_dataset = dataset[b:a]

#Class for treating the graphs dataset as numpy array
#Convenient for converting graph-level information into array
class SliceDataset(Sequence):
    def __init__(self, dataset, idx=0, indices=None):
        self.dataset = dataset
        self.idx = idx
        self.indices = indices

        self.indices_ = (self.indices if self.indices is not None
                         else np.arange(len(self.dataset)))
        self.ndim = 1

    def __len__(self):
        return len(self.indices_)

    @property
    def shape(self):
        return (len(self),)

    def transform(self, data):
        """Additional transformations on ``data``.

        Note: If you use this in conjuction with PyTorch
        :class:`~torch.utils.data.DataLoader`, the latter will call
        the dataset for each row separately, which means that the
        incoming ``data`` is a single rows.

        """
        return data

    def _select_item(self, Xn):
        # Raise a custom error message when accessing out of
        # bounds. However, this will only trigger as soon as this is
        # indexed by an integer.
        try:
            if (self.idx == 0):
                return Xn.x
            if (self.idx == 1):
                return Xn.y
            if (self.idx == 2):
                return Xn.m
        except IndexError:
            name = self.__class__.__name__
            msg = ("{} is trying to access element {} but there are only "
                   "{} elements.".format(name, self.idx, len(Xn)))
            raise IndexError(msg)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xn = self.dataset[self.indices_[i]]
            Xi = self._select_item(Xn)
            return self.transform(Xi)

        cls = type(self)
        if isinstance(i, slice):
            return cls(self.dataset, idx=self.idx, indices=self.indices_[i])

        if isinstance(i, np.ndarray):
            if i.ndim != 1:
                raise IndexError("SliceDataset only supports slicing with 1 "
                                 "dimensional arrays, got {} dimensions instead."
                                 "".format(i.ndim))
            if i.dtype == bool:
                i = np.flatnonzero(i)

        return cls(self.dataset, idx=self.idx, indices=self.indices_[i])

    def __array__(self, dtype=None):
        # This method is invoked when calling np.asarray(X)
        # https://numpy.org/devdocs/user/basics.dispatch.html
        X = [self[i] for i in range(len(self))]
        if np.isscalar(X[0]):
            return np.asarray(X)
        return np.asarray([to_numpy(x) for x in X], dtype=dtype)