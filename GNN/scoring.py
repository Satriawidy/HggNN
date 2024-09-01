import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.feature_selection import mutual_info_regression
from process import weights, SliceDataset

num_classes = len(weights)

#Function to obtain network discriminant and various cut
def apply(y_true, y_score):
    n = num_classes
    f_values = 1/weights
    cut = []
    
    multiply = f_values*y_score + 1e-8
    summed = sum(multiply[:,i] for i in range(n))
    D_val = torch.log(multiply[:,-1]/(summed - multiply[:,-1]))
    D_val = D_val.cpu().numpy()
    
    fpr, tpr, threshold = roc_curve((y_true == n-1), D_val)
    max_signi = np.max(tpr/np.sqrt(fpr+3e-5))
    cut_signi = threshold[np.argmin((tpr-0.5)**2)]
    cut.append(threshold[np.argmin((tpr-0.4)**2)])
    cut.append(threshold[np.argmin((tpr-0.5)**2)])
    cut.append(threshold[np.argmin((tpr-0.6)**2)])
    cut.append(threshold[np.argmin((tpr-0.75)**2)])

    return max_signi, cut_signi, D_val, cut

#Class for computing Jensen-Shannon divergence given P, Q input
class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

JSDD = JSD()

#Function to compute inverse JSD by converting mass and discriminant into
#P, Q and feeding them into JSD class
def JSDinv(mask, massval, D_val, cut_signi):
    #Constructing the mass distribution for
    #accepted and rejected background
    mask_accept = (D_val.squeeze() > cut_signi) & mask
    mask_reject = (D_val.squeeze() <= cut_signi) & mask

    mass_accept = massval[mask_accept]
    mass_reject = massval[mask_reject]


    p1, p2 = np.histogram(mass_accept, bins=50, range=(50,300))
    q1, q2 = np.histogram(mass_reject, bins=50, range=(50,300))

    P = p1/np.sum(p1)
    Q = q1/np.sum(q1)

    return 1/JSDD(F.softmax(torch.tensor(P),dim=0), F.softmax(torch.tensor(Q),dim=0))

#Function to compute AUC value
def aucval(y_true, y_score):
    fpr, tpr, threshold = roc_curve((y_true == len(weights)-1), 
                                    y_score[:,-1])
    auc_value = auc(fpr, tpr)
    return auc_value

#Function to compute acceptance rate from confusion matrix
def accept(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    return cm[-1][-1]/cm.sum(axis=1)[-1]

#Function to compute rejection rate from confusion matrix
def reject(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    
    return 1 - (cm.sum(axis=0)[-1]-cm[-1][-1])/(cm.sum(axis=None)
                - cm.sum(axis=1)[-1])


#Defining scoring system for each-epoch training
def accept_epoch(net, X, y):
    y_pred = net.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return cm[-1][-1]/cm.sum(axis=1)[-1]

def reject_epoch(net, X, y):
    y_pred = net.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return 1 - (cm.sum(axis=0)[-1]-cm[-1][-1])/(cm.sum(axis=None)-cm.sum(axis=1)[-1])