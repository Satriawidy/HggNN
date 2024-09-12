import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import mplhep as hep
from sklearn.metrics import confusion_matrix, auc, roc_curve
from process import weights

def bkgshapes(mass, masked, mask_array, legends, label):
    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV," + "\n" 
                                + "anti-$k_T$, $R=1.0$ jets,"
                                + "\n" +
                                r"$p_T \geq 250$ GeV, $m \geq 50$ GeV" 
                                ), fontsize=18)
    
    x, y = np.histogram(mass[masked], bins=100, range=(50, 210))
    hep.histplot(x, y, histtype="step", label = 'Original', 
                density=True, linewidth=1.75)
    i = 0
    for mask in mask_array:
        x, y = np.histogram(mass[mask], bins=100, range=(50, 210))
        hep.histplot(x, y, histtype="step", label = legends[i], 
                    density=True, linewidth=1.75, linestyle='dashed')
        i = i+1
        
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Jet Fraction')
    plt.xlim(50,210)
    plt.ylim(1e-3, 3e-1)
    plt.yscale('log')
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/Background_shapes_{label}.pdf')
    plt.close()

def bkgindivi(mass, masked, mask_array, legends, label, label_2, label_3):
    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV," + "\n" 
                                + "anti-$k_T$, $R=1.0$ jets,"
                                + "\n" +
                                r"$p_T \geq 250$ GeV, $m \geq 50$ GeV" 
                                + "\n" + "\n" +
                                f"{label_2}"), fontsize=18)
    x, y = np.histogram(mass[masked], bins=100, range=(50, 210))
    hep.histplot(x, y, histtype="step", label = 'Original', 
                density=True, linewidth=1.75)
    i = 0
    for mask in mask_array:
        x, y = np.histogram(mass[mask], bins=100, range=(50, 210))
        hep.histplot(x, y, histtype="step", label = legends[i], 
                    density=True, linewidth=1.75, linestyle='dashed')
        i = i+1
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Jet Fraction')
    plt.xlim(50,210)
    plt.ylim(1e-3, 3e-1)
    plt.yscale('log')
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/Background_shapes_{label}_{label_3}.pdf')
    plt.close()
    
def bkgreject(mass, masked, mask_array, legends, label):
    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV," + "\n" 
                                + "anti-$k_T$, $R=1.0$ jets,"
                                + "\n" +
                                r"$p_T \geq 250$ GeV, $m \geq 50$ GeV" 
                                ), fontsize=18)
    i = 0
    total = mass[masked]
    x, y  = pd.qcut(total, 20, labels=False, retbins=True)
    x, z  = np.histogram(total, bins=y, range=(50, 250))
    v = 1
    
    for mask in mask_array:
        w = mass[mask]
        x1, y1 = np.histogram(w, bins=y)
        hep.histplot(1 - x1/x, y1, histtype="step", label = legends[i],
                    linewidth=1.75)
        v = min(v, np.min(1 - x1/x))
        i = i+1
        
    plt.xlabel('Mass [GeV]')
    plt.ylabel('Background Rejection')
    plt.xlim(50,250)
    plt.ylim(2*v - 1, 2.2 - v)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'figures/Rejection_mass_{label}.pdf')
    plt.close()

def bkgptrans(pT, masked, mask_array, legends, label):
    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV," + "\n" 
                                + "anti-$k_T$, $R=1.0$ jets,"
                                + "\n" +
                                r"$p_T \geq 250$ GeV, $m \geq 50$ GeV" 
                                ), fontsize=18)
    i = 0
    
    total = pT[masked]
    x, y  = pd.qcut(total, 5, labels=False, retbins=True)
    x, z  = np.histogram(total, bins=y, range=(250, 1000))
    
    xx, yy = pd.qcut(total, 100, labels=False, retbins=True)
    xx, zz = np.histogram(total, bins=yy)

    v = 1
    for mask in mask_array:
        w = pT[mask]
        
        x1, y1 = np.histogram(w, bins=y)
        xx1, yy1 = np.histogram(w, bins=yy)
        
        y1[-1] = 1000
        
        err = []
        for j in range(5):
            err.append(np.std((xx/xx1)[j*20:j*20+20]))
        
        hep.histplot(x/x1, y1, yerr = err, xerr = True,
                    histtype='errorbar', label = legends[i], linewidth=1.75)
        v = max(v, np.max(x/x1))
        i = i+1

    plt.xlabel(r'$p_T$ [GeV]')
    plt.ylabel('Inverse Background Efficiency')
    plt.xlim(250,1000)
    plt.ylim(0, v*1.9)
    plt.xscale('log')
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/Rejection_pT_{label}.pdf')
    plt.close()

def bkgetaabs(eta, masked, mask_array, legends, label):
    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV," + "\n" 
                                + "anti-$k_T$, $R=1.0$ jets,"
                                + "\n" +
                                r"$p_T \geq 250$ GeV, $m \geq 50$ GeV" 
                                ), fontsize=18)
    i = 0
    
    total = eta[masked]
    x, y  = pd.qcut(total, 5, labels=False, retbins=True)
    x, z  = np.histogram(total, bins=y, range=(0, 2))
    
    xx, yy = pd.qcut(total, 100, labels=False, retbins=True)
    xx, zz = np.histogram(total, bins=yy)

    v = 1
    for mask in mask_array:
        w = eta[mask]
        
        x1, y1 = np.histogram(w, bins=y)
        xx1, yy1 = np.histogram(w, bins=yy)
        err = []
        
        for j in range(5):
            err.append(np.std((xx/xx1)[j*20:j*20+20]))

        hep.histplot(x/x1, y1, yerr = err, xerr = True,
                    histtype='errorbar', label = legends[i], linewidth=1.75)
        v = max(v, np.max(x/x1))
        i = i+1

    plt.xlabel(r'$|\eta|$')
    plt.ylabel('Inverse Background Efficiency')
    plt.xlim(0,2)
    plt.ylim(0,v*1.9)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/Rejection_eta_{label}.pdf')
    plt.close()

def roc_plot(y_true, preds_, legends, label, name):
    fpr, tpr, threshold = roc_curve((y_true == len(weights)-1),preds_)
    auc_value = auc(fpr, tpr)
    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV, "
                                + "anti-$k_T$, $R=1.0$ jets,"
                                + "\n" +
                                r"$p_T \geq 250$ GeV, $m \geq 50$ GeV" 
                                + "\n" + "\n" +
                                f"{name}, AUC = {auc_value:.3f}"),
                        fontsize=18)
    for i in range(len(weights)-1):
        mask = ((y_true == len(weights) - 1) | (y_true == i))
        y_bkg = y_true[mask]
        pred_ = preds_[mask]
        fpr, tpr, threshold = roc_curve((y_bkg == len(weights)-1), 
                                        pred_)
        plt.plot(tpr,1/(fpr+1e-8), label = legends[i], linestyle = 'dashed',
                linewidth = 1.75)
    if label == 'M016':
        lim = 1e7
    else:
        lim = 1e5            
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Inverse Background Efficiency")
    plt.ylim(1, lim)
    plt.yscale('log')
    plt.grid(visible=True, linestyle = ':', linewidth=0.05)
    plt.legend(loc='upper right', fontsize=18, frameon=False)
    plt.tight_layout()
    plt.savefig(f'figures/ROC_{label}.pdf')
    plt.close()

def auc_plot(y_true, preds, legends, label):
    plt.clf()
    plt.figure(figsize=(8,6))
    i = 0
    for pred in preds:
        fpr, tpr, threshold = roc_curve((y_true == len(weights)-1), pred[:,-1])
        auc_value = auc(fpr, tpr)
        plt.plot(tpr,fpr,label='%s tagger, AUC = %.1f%%'%(legends[i],auc_value*100.))
        i += 1
    plt.semilogy
    plt.ylim(0.001, 1)
    plt.grid(True)
    plt.title('ROC curve', fontsize=25)
    plt.xlabel("sig. efficiency", fontsize=20)
    plt.ylabel("bkg. mistag rate", fontsize=20)
    plt.legend(loc='upper left', fontsize=15, frameon=False)
    plt.savefig(f'figures/AUC_{label}.png')
    plt.close()


def confusion(y_true, y_pred, legends, label):
    plt.clf()
    plt.style.use('default')
    cm = confusion_matrix(y_true, y_pred, normalize= 'true')

    fig, ax = plt.subplots(figsize=(12,10))
    sns.set(font_scale=1.5)
    plt.tick_params(axis='both', labelsize=20)
    # sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['top->bW->bqq', 'Z->qq', 'W->qq', 'H->gg'], yticklabels=['top->bW->bqq', 'Z->qq', 'W->qq', 'H->gg'])
    sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=legends, yticklabels=legends, annot_kws={"fontsize":30})
    plt.xlabel('Predicted Label', loc='right', fontsize=25)
    plt.ylabel('Truth Label', loc='top', fontsize=25)
    plt.tight_layout()
    plt.savefig(f'figures/Confusion_{label}.pdf')
    plt.close()
    
def discplot(D_val, labels, legends, label, cut, legend_bkcut, colors):
    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV," + "\n"
                                + "anti-$k_T$, $R=1.0$ jets,"
                                + "\n" +
                                r"$p_T \geq 250$ GeV," + "\n" 
                                + r"$m \geq 50$ GeV" 
                                ), fontsize=18)
    i = 0
    z = 0
    for legend in legends:
        x, y = np.histogram(D_val[labels==i], bins=100)
        hep.histplot(x, y, histtype="step", label = legend, linewidth=1.75, 
                    density=True)
        z = max(100*np.max(x)/(len(D_val[labels==i]) * 
                              (max(D_val) - min(D_val))), z)
        i = i+1
        
    if label == '0007':
        zli = 1.2
        xmi = -6
        xma = 1.2
    else:
        zli = 1.5
        xmi = -7
        xma = 2.5
    
    for (cu, legend, color) in zip(cut, legend_bkcut, colors):
        plt.vlines(x=cu, ymin=0, ymax=z*zli, ls=':', lw=2, label=legend,
                   colors = color)
                   
    plt.xlabel(r'$D_{Hgg}$')
    plt.ylabel('Jet Fraction')
    plt.ylim(0, z*2.4)
    plt.xlim(xmi, xma)
    plt.legend(ncol=2, columnspacing=0.1, fontsize=18)
    plt.tight_layout()
    plt.savefig(f'figures/Discriminant_{label}.pdf')
    plt.close()
    