import torch
import h5py
import torch.nn.functional as F
import numpy as np
import pandas as pd
from plots import bkgreject, confusion, auc_plot, discplot, bkgindivi, bkgshapes, bkgptrans, bkgetaabs, roc_plot
from scoring import JSDinv, apply
from sklearn.metrics import auc, roc_curve
from sklearn.feature_selection import mutual_info_regression
from process import weights, test_dataset, SliceDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

#Function for converting score and discriminant from h5files
#to graphs and numbers 
def evalplot(model_name, model_legend, label):
    #Label for each jet class in plots
    legend_label = [r'QCD jets', r'Top jets', 
                    r'$H\rightarrow b\bar{b}$', 
                    r'$Z/W\rightarrow$ jets', r'$H\rightarrow gg$']
    #Label for each jet class in filename
    legend_title = ['QCD', 'top', 'Hbb', 'ZW', 'Hgg']
    #Label for each operating point cut in plots
    legend_bkcut = [r'$\epsilon_{Hgg} = 0.40$', 
                    r'$\epsilon_{Hgg} = 0.50$',
                    r'$\epsilon_{Hgg} = 0.60$', 
                    r'$\epsilon_{Hgg} = 0.75$']
    #Color for each
    colors_bkcut = ['blue', 'orange', 'green', 'red']

    #Construct dataloader to extract graph level information quickly
    loader = DataLoader(test_dataset, batch_size=512, shuffle=False,
                        num_workers=16)
    
    #Extracting the graph level information
    y_test_ = []
    massval = []
    ptraval = []
    etabval = []
    for data in tqdm(loader):
        labels, mass, pT, eta = data.y, data.m, data.p, data.e
        y_test_.append(labels.numpy())
        massval.append(mass.numpy())
        ptraval.append(pT.numpy())
        etabval.append(abs(eta.numpy()))
    
    y_test_ = np.concatenate(y_test_, axis=0)
    massval = np.concatenate(massval, axis=0)
    ptraval = np.concatenate(ptraval, axis=0)
    etabval = np.concatenate(etabval, axis=0)
    
    
    mask_acc = []
    mask_eac = []
    preds = []
    
    #Masking to only get backgrounds
    mask = (y_test_.squeeze() != len(weights) - 1)
    
    for (name, legend) in zip(model_name, model_legend):
        with h5py.File(f'results/result_{name}.h5', 'r') as f:
            P = pd.DataFrame(f['preds'])
            S = pd.DataFrame(f['score'])
        
        y_score = S.values
        y_preds = P.values
            
        fpr, tpr, threshold = roc_curve((y_test_ == len(weights)-1),
                                        y_score[:,-1])
        auc_value = auc(fpr, tpr)
            
        max_signi, cut_signi, D_val, cut = apply(y_test_, y_score)
        JSDval = JSDinv(mask, massval, D_val, cut_signi)
        mutual = mutual_info_regression(massval[mask].reshape(-1, 1), 
                                        D_val[mask], n_neighbors=3, 
                                        copy=True, random_state=20)
        
        #Printing the quantitative results    
        print(f'AUC = {auc_value}, Max signif. = {max_signi}, Cut at = {cut_signi}')
        print(f'mutual information = {mutual}, inverse JSD = {JSDval}')
            
        #Plot the confusion matrix and discriminant
        confusion(y_test_, y_preds, legend_label, name)
        discplot(D_val, y_test_, legend_label, name,
                 cut, legend_bkcut, colors_bkcut)
        
        #Masking for each remaining background class after cut on
        #50% operating point    
        mask_bkg = []
        mask_bkc = []
        for j in range(len(cut)):
            mask_cut = (D_val.squeeze() > cut[j])
            mask_bkg.append(mask & mask_cut)
            mask_bkc.append(mask_cut)
        
        #Plotting various plot with separated class, for each model
        bkgshapes(massval,mask,mask_bkg,legend_bkcut,f'{name}')
        bkgreject(massval,mask,mask_bkg,legend_bkcut,f'{name}')
        bkgptrans(ptraval,mask,mask_bkg,legend_bkcut,f'{name}')
        bkgetaabs(etabval,mask,mask_bkg,legend_bkcut,f'{name}')
        roc_plot(y_test_, D_val, legend_label, f'{name}', f'{legend}')
        
        #Plotting mass shape for each class, for each model    
        j = 0
        for legend in legend_title:
            mask_each = []
            mask_ones = (y_test_.squeeze() == j)
            k = 0
            for c in cut:
                mask_each.append(mask_ones & mask_bkc[k])
                k += 1
            bkgindivi(massval, mask_ones, mask_each, legend_bkcut, 
                      f'{name}', legend_label[j], legend)
            j += 1

        mask_cut = (D_val.squeeze() > cut_signi)
        mask_acc.append(mask & mask_cut)
        mask_eac.append(mask_cut)
            
    #Plotting various plot with separated class, for all model
    bkgshapes(massval, mask, mask_acc, model_legend, label)
    bkgreject(massval, mask, mask_acc, model_legend, label)
    bkgptrans(ptraval, mask, mask_acc, model_legend, label)
    bkgetaabs(etabval, mask, mask_acc, model_legend, label)
    auc_plot(y_test_, preds, model_legend, label)
    
    #Plotting mass shape for each class, for all model    
    j = 0
    for legend in legend_title:
        mask_each = []
        mask_ones = (y_test_.squeeze() == j)
        k = 0
        for model in model_name:
            mask_each.append(mask_ones & mask_eac[k])
            k += 1
        bkgindivi(massval, mask_ones, mask_each, model_legend, label, 
                  legend_label[j], legend)
        j += 1
