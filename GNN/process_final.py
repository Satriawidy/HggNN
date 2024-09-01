import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import torch
import os.path as osp
import os
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

#This is the same module for processing the data, but used only for final
#evaluation (significance analysis dataset)

hfivesdir = /storage/teaching/SummerProjects/2024/s2601026/Analysis
graphsdir = /localstorage/s2601026/GNN_final
resultdir = /home/s2601026/Dissertation/GNN/result_final

class MyOwnDataset(Dataset):
    def __init__(self, root, direname, filename, transform = None, pre_transform = None, pre_filter = None):
        self.direname = direname
        self.filename = filename
        super().__init__(root, transform, pre_transform, pre_filter)
        
    
    @property
    def raw_file_names(self):
        return 'new_Input_Tagging_v7.h5'
    
    @property
    def processed_file_names(self):
        return 'new_Input_Tagging_v7.pt'
    
    def download(self):
        pass

    def process(self):
        filename = self.filename
        direname = self.direname
        with h5py.File(f"hfivesdir/{direname}/{filename}", 'r') as f:
            keys = [key for key in f.keys()]
            if keys != []:
                #self.data = pd.DataFrame(f[keys[0]]['2d'])
                self.ones = pd.DataFrame(f[keys[0]]['1d'][:])
            else:
                self.ones = pd.DataFrame([])
            
        pass
        
        #print(f"{direname}/{filename}")
        #i = 0
        #for row in self.data.itertuples():
        #    node_feats = self._get_node_features(row, i)
        #    label = self._get_labels(i)

        #    data = Data(x = node_feats,
        #                y = label)
            
        #    if self.pre_transform is not None:
        #         data = self.pre_transform(data)

        #    torch.save(data, osp.join(self.processed_dir, f'new_Input_Tagging_{i}.pt'))
        #    if (i%35000==1):
        #        print(i)
        #    i += 1

    def _get_node_features(self, row, i):
        all_node_feats = []

        for column in row[1:]:
            column_1 = np.asarray([column[3],column[4],column[5],column[6],column[7]])
            column_2 = np.asarray([column[0],column[1],column[2]])
            if np.isnan(column_1[0]) == 0:
                column_3 = [1]
                column_3.extend(column_1)
                nodes = self._preprocess(column_3, i)
                all_node_feats.append(nodes)
            if np.isnan(column_2[0]) == 0:
                column_4 = [0, 0, 0]
                column_4.extend(column_2)
                nodes = self._preprocess(column_4, i)
                all_node_feats.append(nodes)
                
        all_node_feats = np.asarray(all_node_feats)
        
        return torch.tensor(all_node_feats, dtype=torch.float)
    
    def _preprocess(self, column, i):
        E_const = column[5] * np.cosh(column[3])
        p_jet = self.ones['FJet_pT'][i] * np.cosh(self.ones['FJet_Eta'][i])
        m_jet = self.ones['FJet_Mass'][i]
        E_jet = np.sqrt(p_jet**2 + m_jet**2)
        E_pTT = np.log(column[5] / self.ones['FJet_pT'][i])

        column[3] = column[3] - self.ones['FJet_Eta'][i]
        column[4] = column[4] - self.ones['FJet_Phi'][i]
        column[5] = np.log(column[5])

        R_const = np.sqrt(column[3]**2 + column[4]**2)

        add = [E_pTT, np.log(E_const), np.log(E_const / E_jet), R_const]

        nodes = np.append(column, add)
        
        return nodes

    def _get_labels(self, i):
        label = np.asarray(self.ones['FJet_flavour'][i])

        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.ones.shape[0]

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'new_Input_Tagging_{idx}.pt'))
        return data

def dataset(direname, filename):
    data = MyOwnDataset(root=f"graphsdir/{direname}/{filename}", direname=direname, filename=filename)
    return data

#Uncomment for converting h5 dataset into graph dataset

#for filename in os.listdir("hfivesdir/0L/"):
#    if filename not in os.listdir("resultdir/0L/"):
#        data = dataset(direname = '0L', filename = filename)
    
#for filename in os.listdir("hfivesdir/1L/"):
#    if filename not in os.listdir("resultdir/1L/"):
#        data = dataset(direname = '1L', filename = filename)

#for filename in os.listdir("hfivesdir/2L/"):
#    if filename not in os.listdir("resultdir/2L/"):
#        data = dataset(direname = '2L', filename = filename)
