import torch
import os.path as osp
import torch.nn.functional as F
import torch.optim as optim
import scipy.stats as stats
import numpy as np
import pandas as pd

from torch import nn
from tqdm import tqdm

from process import weights, X_train_tensor, y_train_tensor
from module import DNN, DNNMini
from neural import NeuralNetBase, NeuralNetDisCo
from scoring import aucval, accept, reject
from scoring import accept_epoch, reject_epoch

from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skorch.callbacks import EpochScoring, EarlyStopping, Checkpoint
from scipy.stats import rv_discrete

#Function for reporting n best performing network during the search
def report(results, n_top):
    for i in range(1, n_top+1):
        candidates = np.flatnonzero(results["rank_test_auc"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Accuracy: {0:.3f} ({1:.3f})".format(
                  results["mean_test_acc"][candidate],
                  results["std_test_acc"][candidate]),
                  "Rejection: {0:.3f} ({1:.3f})".format(
                  results["mean_test_reject"][candidate],
                  results["std_test_reject"][candidate]),
                  "AUC: {0:.3f} ({1:.3f})".format(
                  results["mean_test_auc"][candidate],
                  results["std_test_auc"][candidate]))
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Constructing callbacks for monitoring the training 
reje_e = EpochScoring(reject_epoch, lower_is_better=False)
acce_e = EpochScoring(accept_epoch, lower_is_better=False)
stop = EarlyStopping(monitor='valid_acc', patience=5, lower_is_better=False, 
                      threshold=-0.001, load_best=True)
cp = Checkpoint(monitor='valid_acc_best')

#Base model for hyperparameter search
model = NeuralNetBase(DNNMini([128, 128, 128], 0.13, 15, len(weights)),
                     criterion = nn.CrossEntropyLoss,
                     criterion__reduction = 'none',
                     criterion__weight = weights,
                     optimizer = optim.Adam,
                     optimizer__lr = 0.0005,
                     batch_size = 256,
                     max_epochs = 20,
                     verbose = 0,
                     device = device,
                     callbacks = [stop, cp]
                     )

#Create scorer for hyperparameter search
rejv = make_scorer(reject, greater_is_better=True)
accv = make_scorer(accept, greater_is_better=True)
aucv = make_scorer(aucval, greater_is_better=True, needs_proba=True)

scoring = {"reject": rejv, "acc": accv, "auc": aucv}


#Array for hidden layers combination in hyperparameter search
sizes = [32, 64, 128, 256]
numbers = [2, 3, 4, 5]
shapes = []
for number in numbers:
    for size in sizes:
        shape = []
        for k in range(number):
            shape.append(size)
        shapes.append(shape)   

#Array for batch sizes distribution in hyperparameter search
values = [j for j in range(64, 2048)]
probas = [1/(j*np.log(2048/64)) for j in range(64, 2048)]
probss = probas/np.sum(probas) 

#Parameter distribution for hyperparameter search
param_dist = {
    'batch_size': rv_discrete(values=(values, probss)),
    'max_epochs': stats.randint(5, 30),
    'optimizer__lr' : stats.loguniform(1e-4, 1e-1),
    'module__dropout_rate' : stats.uniform(0.1, 0.8),
    'module__hidden' : shapes,
    'module__input_dim' : [15],
    'module__output_dim' : [len(weights)]
    }

#Randomised search
#cv = number of cross-validation
#n_jobs = number of cpu used during the search
#verbose = 3 pointing out to how much information is written into terminal
#during the search
#n_iter = number of networks used in the search
random = RandomizedSearchCV(estimator = model, param_distributions = param_dist, n_jobs = 10,
                    cv = 5, verbose = 3, scoring = scoring, 
                    refit = "reject", n_iter=50)

random_result = random.fit(X_train_tensor, y_train_tensor)

#Reporting the 5 best performing network
report(random_result.cv_results_, 5)

#Writing the relevant information about the search into a csv file
df = pd.DataFrame(random_result.cv_results_)
df.to_csv('result_new.csv',mode='w', header=True)



