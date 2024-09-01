import torch
import h5py
import torch.nn.functional as F
import numpy as np
from plots import bkgreject, confusion, auc_plot, discplot, bkgindivi, bkgshapes, bkgptrans, bkgetaabs, roc_plot
from scoring import JSDinv, apply
from sklearn.metrics import auc, roc_curve
from sklearn.feature_selection import mutual_info_regression
from process import weights, test_dataset, SliceDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

#Function for producing h5 files containing discriminant and GNN score
#The files can then be processed into results and graphs.
def model_eval(models, model_name):
    X_test_ = test_dataset
    
    for (model, name) in zip(models, model_name):
        with torch.no_grad():
            y_score = model.predict_proba(X_test_)
            y_preds = model.predict(X_test_)
            
        f_values = 1/weights
        multiply = f_values*y_score + 1e-8
        summed = sum(multiply[:,i] for i in range(len(weights)))
        D_val = torch.log(multiply[:,-1]/(summed - multiply[:,-1]))
        D_val = D_val.cpu().numpy()
            
        hf = h5py.File(f'results/result_{name}.h5', 'w')
        hf.create_dataset('D_val', data=D_val)
        hf.create_dataset('preds', data=y_preds)
        hf.create_dataset('score', data=y_score)
        hf.close()

#Function for producing h5 files containing discriminant and GNN score
#from evaluation (significance analysis) dataset.
def eval_final(model, dataset, direname, filename):
    #Checking if the dataset have entries (not len = 0)
    if len(dataset) != 0:
        with torch.no_grad():
            y_score = model.predict_proba(dataset)
        
        #Construct discriminant
        f_values = 1/weights
        multiply = f_values*y_score + 1e-8
        summed = sum(multiply[:,i] for i in range(len(weights)))
        D_val = torch.log(multiply[:,-1]/(summed - multiply[:,-1]))
        D_val = D_val.cpu().numpy()
        
        #Saving the discriminant and score into h5files
        hf = h5py.File(f'result_final/{direname}/{filename}', 'w')
        hf.create_dataset('D_val', data=D_val)
        hf.create_dataset('score', data=y_score)
        hf.close()
    #If no entries, just put dataset (also no entries)
    else:
        hf = h5py.File(f'result_final/{direname}/{filename}', 'w')
        hf.create_dataset('D_val', data=dataset)
        hf.create_dataset('score', data=dataset)
        hf.close()
    print(f"Done {direname}/{filename}")