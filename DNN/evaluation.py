import torch
import h5py
import torch.nn.functional as F
import numpy as np
from plots import bkgreject, confusion, auc_plot, discplot, bkgindivi, bkgshapes, bkgptrans, bkgetaabs, roc_plot
from scoring import JSDinv, apply
from sklearn.metrics import auc, roc_curve
from sklearn.feature_selection import mutual_info_regression
from process import weights, scaler, cols_inv, X_test_tensor, y_test_tensor
from torch_geometric.loader import DataLoader
from tqdm import tqdm

#Function for producing discriminant and DNN score
#The discriminant and DNN score are then processed into results and plots
#The discriminant and DNN score are also written into h5 files
def model_eval(models, model_name, model_legend, label):
    #Label for each jet class in plots
    legend_label = [r'QCD jets', r'Top jets', 
                    r'$H\rightarrow b\bar{b}$', 
                    r'$Z/W \rightarrow$ jets', r'$H\rightarrow gg$']
    #Label for each jet class in filename
    legend_title = ['QCD', 'top', 'Hbb', 'ZW', 'Hgg']
    #Label for each operating point cut in plots
    legend_bkcut = [r'$\epsilon_{H\rightarrow gg} = 0.40$', 
                    r'$\epsilon_{H\rightarrow gg} = 0.50$',
                    r'$\epsilon_{H\rightarrow gg} = 0.60$', 
                    r'$\epsilon_{H\rightarrow gg} = 0.75$']
    #Color for each operating point cut in plots
    colors_bkcut = ['blue', 'orange', 'green', 'red']
                    
    X_test = scaler.inverse_transform(X_test_tensor[:,cols_inv]) 
    
    #Extracting some jet level information
    X_test_ = X_test_tensor
    y_test_ = y_test_tensor.numpy()
    massval = X_test[:,2]
    ptraval = X_test[:,6]
    etabval = abs(X_test[:,1])
    
    mask_acc = []
    mask_eac = []
    preds = []
    
    #Masking to only get backgrounds
    mask = (y_test_.squeeze() != len(weights) - 1)
    
    i = 0
    for model in models:
        with torch.no_grad():
            y_score = model.predict_proba(X_test_)
            y_preds = model.predict(X_test_)
            
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
            confusion(y_test_, y_preds, legend_label, model_name[i])
            discplot(D_val, y_test_, legend_label, model_name[i],
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
            bkgshapes(massval,mask,mask_bkg,legend_bkcut,f'{model_name[i]}')
            bkgreject(massval,mask,mask_bkg,legend_bkcut,f'{model_name[i]}')
            bkgptrans(ptraval,mask,mask_bkg,legend_bkcut,f'{model_name[i]}')
            bkgetaabs(etabval,mask,mask_bkg,legend_bkcut,f'{model_name[i]}')
            roc_plot(y_test_, D_val, legend_label, f'{model_name[i]}',
                     f'{model_legend[i]}')
            
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
                          f'{model_name[i]}', legend_label[j], legend)
                j += 1

            mask_cut = (D_val.squeeze() > cut_signi)
            mask_acc.append(mask & mask_cut)
            mask_eac.append(mask_cut)
            
            preds.append(y_score)
            
            #Saving the discriminant and score into h5files
            hf = h5py.File(f'result_29/result_{model_name[i]}.h5', 'w')
            hf.create_dataset('D_val', data=D_val)
            hf.create_dataset('truth', data=y_test_)
            hf.create_dataset('score', data=y_score)
            hf.close()
            
            i += 1
            
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
        for model in models:
            mask_each.append(mask_ones & mask_eac[k])
            k += 1
        bkgindivi(massval, mask_ones, mask_each, model_legend, label, 
                  legend_label[j], legend)
        j += 1