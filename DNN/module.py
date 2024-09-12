import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim

from sklearn.preprocessing import MinMaxScaler
from process import cols_dnn
from torch import nn

class DNN(nn.Module):
    def __init__(self, hidden, dropout_rate, input_dim, output_dim):
        super(DNN, self).__init__()
        
        self.hidden = nn.ModuleList()

        #Input layer
        self.hidden.append(nn.Linear(input_dim, hidden[0]))
        
        #Hidden layers
        for i in range(len(hidden)-1):
            self.hidden.append(nn.Linear(hidden[i], hidden[i+1]))
            self.hidden.append(nn.BatchNorm1d(hidden[i+1]))
            self.hidden.append(nn.ReLU())
        
        #Activation function (only use ReLU at the end)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

        #Dropout layer  
        self.dropout = nn.Dropout(dropout_rate)
        
        #Output layer
        self.out = nn.Linear(hidden[-1], output_dim)
    
    def forward(self, x):

        #Forward the data through the input and hidden layers
        for layer in self.hidden:
            x = layer(x)
        
        #Applying the dropout before the output layer
        x = self.dropout(x)    
        #Output layer
        x = self.out(x)

        return x

class DNNMini(DNN):
    def forward(self, x):
        y = x[:, cols_dnn] 

        #Forward the data through the input and hidden layers
        for layer in self.hidden:
            y = layer(y)
        
        #Applying the dropout before the output layer
        y = self.dropout(y)    
        #Output layer
        y = self.out(y)

        return y