import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from torch import nn
from collections.abc import Sequence
from skorch.utils import to_numpy 

pd.options.mode.chained_assignment = None  # default='warn'

hfivesdir = r"/storage/teaching/SummerProjects/2024/s2601026/input/new_Input_Tagging_29thJuly2024.h5"

#Open the h5 file containing jet kinematic information
with h5py.File(hfivesdir) as f:
    df = pd.DataFrame(f['LargeRJet']['1d'][:])

#Change the flavour/class label
df.replace({'flavour':{0:0, 6:1, 10:2, 23:3, 24:3, 42:4}}, inplace=True)    
df.dropna(inplace=True)

#Set the flavour label as the truth record for training
X = df.drop(['flavour'], axis=1)
y = df['flavour']

#Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1234)

X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)

#Normalise the dataset values for convenience
cols_to_normalize = X_train.select_dtypes(include='number').columns.to_list()
cols_to_normalize.remove('weight')
cols_to_normalize.remove('Eta_Bin')
cols_to_normalize.remove('pT_Bin')

scaler = MinMaxScaler()
scaler.fit(X_train[cols_to_normalize])

X_train[cols_to_normalize] = scaler.transform(X_train[cols_to_normalize])
X_test[cols_to_normalize] = scaler.transform(X_test[cols_to_normalize])

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

#The columns that will be needed for inversing the normalisation
cols_inv = [0,2,3,4,5,6,7,9,10,11,12,13,14,15,16]
#The columns that will be used as the network input
cols_dnn = [0,1,3,4,5,6,8,9,10,11,12,13,14,15,16]

#Create the class weight for each flavour
value_counts = df['flavour'].value_counts()
total_count = len(df)
weights = total_count / value_counts.sort_index().values
weights = torch.tensor(weights, dtype=torch.float)

