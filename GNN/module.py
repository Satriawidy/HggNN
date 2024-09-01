import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx
import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.transforms import KNNGraph
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from torch_geometric.loader import DataLoader
from torch import nn
from torch_geometric.nn import EdgeConv, global_mean_pool
from torch_cluster import knn_graph

#Module for EdgeConvBlock of ParticleNet. The code is taken from
#https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/6.JetTaggingGCN.ipynb
class EdgeConvBlock(nn.Module):
    def __init__(self, in_size,layer_size):
        super(EdgeConvBlock, self).__init__()
        
        layers = []
        layers.append(nn.Linear(in_size * 2, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for i in range(2):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    
    def __repr__(self):
        return "{}(nn={})".format(self.__class__.__name__, self.model)

#Module for ParticleNet. The code is modified from
#https://github.com/makagan/SSI_Projects/blob/main/jet_notebooks/6.JetTaggingGCN.ipynb
class ParticleNet(nn.Module):
    def __init__(self, kernel_sizes, fc_size, dropout, k, node_feat_size, num_classes=6):
        super(ParticleNet, self).__init__()
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes

        self.k = k
        self.num_edge_convs = len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.fc_size = fc_size
        self.dropout = dropout

        self.edge_nets = nn.ModuleList()
        self.edge_convs = nn.ModuleList()

        self.kernel_sizes.insert(0, self.node_feat_size)
        self.output_sizes = np.cumsum(self.kernel_sizes)

        self.edge_nets.append(EdgeConvBlock(self.node_feat_size, self.kernel_sizes[1]))
        self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr="mean"))

        for i in range(1, self.num_edge_convs):
            self.edge_nets.append(EdgeConvBlock(self.output_sizes[i], self.kernel_sizes[i+1]))
            self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr="mean"))
        
        self.fc1 = nn.Sequential(nn.Linear(self.output_sizes[-1], 
                                self.fc_size))
        self.dropout_layer = nn.Dropout(p = self.dropout)
        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

    def forward(self, data):
        x = data.x
        batch = data.batch
        
        #Extracting the Delta phi and Delta eta for initial node position
        pos = data.x[:,[3,4]] 

        for i in range(self.num_edge_convs):
            edge_index = (knn_graph(pos, self.k, batch) if i==0 else
            knn_graph(x, self.k, batch))

            x = torch.cat((self.edge_convs[i](x, edge_index), x), dim=1)
        
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.dropout_layer(x)

        return self.fc2(x)

#Module for ParticleNet if there are pTbin/etabin and we want to ignore them
class ParticleNetMini(nn.Module):
    def __init__(self, kernel_sizes, fc_size, dropout, k, node_feat_size, num_classes=6):
        super(ParticleNetMini, self).__init__()
        self.node_feat_size = node_feat_size
        self.num_classes = num_classes

        self.k = k
        self.num_edge_convs = len(kernel_sizes)
        self.kernel_sizes = kernel_sizes
        self.fc_size = fc_size
        self.dropout = dropout


        self.edge_nets = nn.ModuleList()
        self.edge_convs = nn.ModuleList()

        self.kernel_sizes.insert(0, self.node_feat_size)
        self.output_sizes = np.cumsum(self.kernel_sizes)

        self.edge_nets.append(EdgeConvBlock(self.node_feat_size, self.kernel_sizes[1]))
        self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr="mean"))

        for i in range(1, self.num_edge_convs):
            self.edge_nets.append(EdgeConvBlock(self.output_sizes[i], self.kernel_sizes[i+1]))
            self.edge_convs.append(EdgeConv(self.edge_nets[-1], aggr="mean"))
        
        self.fc1 = nn.Sequential(nn.Linear(self.output_sizes[-1], 
                                self.fc_size))
        self.dropout_layer = nn.Dropout(p = self.dropout)
        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

    def forward(self, data):
        x = data.x[:,[0,1,2,3,4,5,6,7,8,9]]
        batch = data.batch
        pos = data.x[:,[3,4]] 

        for i in range(self.num_edge_convs):
            edge_index = (knn_graph(pos, self.k, batch) if i==0 else
            knn_graph(x, self.k, batch))

            x = torch.cat((self.edge_convs[i](x, edge_index), x), dim=1)
        
        x = global_mean_pool(x, batch)
        x = self.fc1(x)
        x = self.dropout_layer(x)

        return self.fc2(x)