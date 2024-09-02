import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score
import pickle
import mplhep as hep

hfivesdir = '/storage/teaching/SummerProjects/2024/s2601026/Analysis'
hfiveedir = '/home/s2601026/Dissertation/GNN/result_final'
modelsdir = '/home/s2601026/Dissertation/GNN/models_bdt'
figuredir = '/home/s2601026/Dissertation/GNN/figure_bdt'


#-------------------------Initialisation------------------------------#
#Initialising the filename for each class of events
name_jet = ['Zjets', 'Wjets']
name_top = ['ttbar']
name_hbb = ['WmHbb', 'WpHbb', 'ZHbb', 'ggZHbb']
name_dib = ['WW', 'WZ', 'ZZ']
name_hgg = ['ZHgg', 'WpHgg', 'WmHgg', 'ggZHgg']

names = [name_hgg, name_hbb, name_dib, name_top, name_jet]
labels = ['Hgg', 'Hbb', 'ZW', 'Top', 'QCD']

#Putting the QCD jet, top, and diboson event into one category
#for BDT training (later)
names_inc = [name_hgg, 
             name_hbb, 
             list(np.concatenate([name_jet, name_top, 
                                  name_dib]))]
label_inc = ['Hgg', 'Hbb', 'Background']


#----------------------Split Into Train-Test-------------------------#
#Function for splitting the j-lepton channel into signal-background and
#train-test dataset
def split(j):
    events = []
    for name in names:
        event = pd.DataFrame()
        for filename in name:
            with h5py.File(f"{hfivesdir}/{j}L/{filename}_{j}L.h5",'r') as f:
                keys = [key for key in f.keys()]
                #Only do this procedure if the key exist (dataset is not None)
                if keys != []:
                    dg = pd.DataFrame(f[keys[0]]['1d'][:])
                    #Open the corresponding h5 files containing GNN discriminant
                    with h5py.File(f"{hfiveedir}/{j}L/{filename}_{j}L.h5",'r') as f:
                        dh = pd.DataFrame(f['D_val'])
                        dh.columns = ['D_val']
                        df = pd.concat([dg, dh], axis=1)
                        event = pd.concat([event, df], axis = 0)
        #Perform common selection on the event
        event = event[event['nBTaggedJets'] == 0]
        event = event[event['nTauTaggedJets'] == 0]
        
        #Perform channel-exclusive selection on the event
        if j==1:
            event = event[event['pTW'] > 250]
        elif j==2:
            event = event[event['pTZ'] > 250]
        events.append(event)
    
    X = pd.concat(events, axis=0)
    #Constructing signal (1) and background (0) label for training
    y = np.concatenate((np.ones(events[0].shape[0]), 
                        np.ones(events[1].shape[0]),
                        np.zeros(sum(events[i].shape[0] for i in range(2,5)) )))
    #Constructing class label (0-4) for testing and stack-plotting
    z = np.concatenate([np.full(events[i].shape[0], 
                                i) for i in range(len(names))])
    #Splitting the dataset into two equal part
    X_train,X_test_, y_train,y_test_, z_train, z_test_ = train_test_split(X, y, z,
                                              test_size=0.5, random_state=42)
    #Recovering the lumi_weight and discriminant
    train_weight = X_train['Lumi_weight']
    test__weight = X_test_['Lumi_weight']
    dval_train = X_train['D_val']
    dval_test_ = X_test_['D_val']
    #Removing variables that are not used for BDT training
    X_train = X_train.drop(['FJet_Phi', 'FJet_flavour', 'Lumi_weight',
                             'D_val', 'nBTaggedJets', 'nTauTaggedJets',
                               'NFJets'], axis=1)
    X_test_ = X_test_.drop(['FJet_Phi', 'FJet_flavour', 'Lumi_weight', 
                          'D_val', 'nBTaggedJets', 'nTauTaggedJets', 
                          'NFJets'], axis=1)

    train_weight.reset_index(drop=True, inplace=True)
    test__weight.reset_index(drop=True, inplace=True)
    dval_train.reset_index(drop=True, inplace=True)
    dval_test_.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_test_.reset_index(drop=True, inplace=True)

    train_weight = train_weight.values
    test__weight = test__weight.values
    dval_train = dval_train.values
    dval_test_ = dval_test_.values
    X_train = X_train.values
    X_test_ = X_test_.values

    return X_train, X_test_, y_train, y_test_, train_weight, test__weight, z_train, z_test_, dval_train, dval_test_


#-------------------------Training------------------------------#
#Function for training the BDT model
def train_bdt(j, params, name):
    X_train, X_test_, y_train, y_test_, train_weight, test__weight, z_train, z_test_, dval_train, dval_test_ = split(j)
    
    #Setting negative weight to small non-zero values
    train_weight[train_weight<0] = 0.0001
    test__weight[test__weight<0] = 0.0001

    weight_train = train_weight
    weight_test_ = test__weight

    #Setting the value for min_samples_split and min_samples_leaf
    numbers = int(0.005*len(X_train))

    #Initialising the model A with specified params
    model_train = GradientBoostingClassifier(**params, 
                                             min_samples_split = numbers, 
                                             min_samples_leaf = numbers)
    #Training the model A
    model_train.fit(X_train, y_train, sample_weight=weight_train)
    
    #Saving the trained model A
    with open(f'{modelsdir}/model_train_{name}_{j}.pkl','wb') as f:
        pickle.dump(model_train,f)

    #Repeating the same procedure for model B
    model_test_ = GradientBoostingClassifier(**params, 
                                            min_samples_split = numbers, 
                                            min_samples_leaf = numbers)
    model_test_.fit(X_test_, y_test_, sample_weight=weight_test_)
    
    with open(f'{modelsdir}/model_test__{name}_{j}.pkl','wb') as f:
        pickle.dump(model_test_,f)

#-------------------------Evaluation------------------------------#
#Function for cross-evaluating the models 
def test_bdt(j, name):
    X_train, X_test_, y_train, y_test_, train_weight, test__weight, z_train, z_test_, dval_train, dval_test_ = split(j)

    #Opening both model
    with open(f'{modelsdir}/model_train_{name}_{j}.pkl', 'rb') as f:
        model_train = pickle.load(f)
    with open(f'{modelsdir}/model_test__{name}_{j}.pkl', 'rb') as f:
        model_test_ = pickle.load(f)
    
    #Cross-Evaluation
    score_train = model_test_.decision_function(X_train)
    score_test_ = model_train.decision_function(X_test_)

    #Grouping the necessary variables (label, BDT score, lumi_weight, discriminant)
    y = np.concatenate((z_train, z_test_))
    scores = np.concatenate((score_train, score_test_))
    weight = np.concatenate((train_weight, test__weight))
    dvalue = np.concatenate((dval_train, dval_test_))

    #Reparametrising the score to fit in [-1,1]
    scores = np.tanh((scores - (max(scores) + min(scores))/2))

    return y, scores, weight, dvalue


#-------------------------Significance------------------------------#
#Function for computing squared Asimov significance 
def sign(S, B, sigma):
    er = (sigma*B + 1e-10)**2
    Z1 = (S + B)*np.log((S + B + 1e-10)*(B + er) / 
                        (B**2 + (S + B)*(er) + 1e-10))
    Z2 = (B**2/er)*np.log(1 + S*er/(B*(B + er) + 1e-10))
    ZZ = 2*(Z1 - Z2)
    
    return ZZ

#Function for computing the total significance for all bins in a channel
def sign_bin(SS, BB, sigma, certain=True):
    sign2 = 0.0
    #Compute significance in each bin and sum over
    for S,B in zip(SS, BB):
        #To prevent NaN value when B=0
        if B == 0:
            sign2 += 0
        else:
            #Certain means zero background uncertainty
            #Uncertain means otherwise
            if certain==True:
                sign2 += (S**2)/B
            else:
                sign2 += sign(S, B, sigma)
    return np.sqrt(sign2)

#---------------------------Plotting--------------------------------#
#Function to plot the BDT score distribution (stacked and unstacked)
def plotbin(y, scores, weight, dvalue, j, binned, cut, ylim):
    #Differentiate between signals and backgrounds
    signal = scores[(dvalue > cut) & (y == 0)]
    backgr = scores[(dvalue > cut) & (y != 0)]
    sweigh = weight[(dvalue > cut) & (y == 0)]
    bweigh = weight[(dvalue > cut) & (y != 0)]

    #Performing cut on GNN discriminant
    events = [scores[(dvalue > cut) & 
                     (y == i)] for i in range(len(names))]
    eweigh = [weight[(dvalue > cut) & 
                     (y == i)] for i in range(len(names))]

    numbin = len(binned)

    #Divide the signals and backgrounds into specified bins
    a, b = np.histogram(backgr, weights=bweigh, bins= binned)
    x, b = np.histogram(signal, weights=sweigh, bins= binned)

    #Defining max value for the z axis to fit the plot
    z = numbin*max(max(a)/sum(a), max(x)/sum(x))
    print(z, max(x)/sum(x), max(a)/sum(a))
    
    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV" + 
                                f", {j}-lepton channel," + "\n"
                                + r"$p_T^V > 250$ GeV, " + 
                                r"$D_{Hgg}\geq$" + f"{cut}"))
    hep.histplot(x, b, histtype="step", label = "Signal", 
                 density=True, linewidth=2)
    hep.histplot(a, b, histtype="step", label = "Background", 
                 density=True, linewidth=2)
    plt.legend()
    plt.xlabel('BDT Score')
    plt.ylabel('Event Fraction')
    plt.ylim(0, z/1.3)
    plt.savefig(f'{figuredir}/Result_a_{j}.pdf')
    plt.show()

    flavour_map = {
            0: (r'$H\rightarrow gg$', 'k'),
            1: (r'$H\rightarrow b\bar{b}$', 'r'),
            2: ('Diboson', 'g'),
            3: (r'$t\bar{t}$', 'y'),
            4: (r'$W/Z$ + jets', 'b')
        }

    X = []
    B = []
    C = []
    L = []
    #Constructing the array containing number of events for each class
    #Together with specifying their color and label
    for key, (label, color) in flavour_map.items():
        x, b = np.histogram(events[key], weights=eweigh[key], 
                            bins=binned)
        X.append(x)
        B.append(b)
        C.append(color)
        L.append(label)


    plt.style.use('default')
    plt.clf()
    plt.figure(figsize=(8,6))
    hep.style.use("ATLAS")
    hep.label.exp_label(exp='Simulation', data=True, loc=4, 
                        rlabel=(r"$\sqrt{s}$ = 14 TeV, 3000 fb$^{-1}$,"
                                 + "\n" + 
                                f"{j}-lepton channel, " + 
                                r"$p_T^V > 250$ GeV," + "\n" +
                                r"$D_{Hgg}\geq $" + f"{cut}"))
    hep.histplot(X, b, histtype="fill", label=L, color=C, stack=True)
    plt.legend()
    plt.xlabel('BDT Score')
    plt.ylabel('Events/0.125')
    plt.yscale('log')
    plt.ylim(1e-1, ylim)
    plt.savefig(f'{figuredir}/Result_b_{j}.pdf')

    plt.show()

#-------------------------Main Training------------------------------#
#Initialising the parameters for training BDT
params_1 = {'loss':'exponential', 'learning_rate':0.1, 
            'n_estimators':600, 'max_depth':8, 'verbose':10,
            'subsample':0.25}

##Uncomment these lines to train model A and model B for all channels with params_1
#train_bdt(0, params_1, 1)
#train_bdt(1, params_1, 1)
#train_bdt(2, params_1, 1)

#-------------------------Main Evaluation----------------------------#
#Examples for evaluating the BDT on 0-lepton channel with model using params_1
y, scores, weight, dvalue = test_bdt(0, 1)

#Defining the cut position on GNN discriminant
cut = 0.9
#Dividing the BDT scores and lumi_weight for signal and backgrounds
fscores_signal = scores[(dvalue > cut) & (y == 0)]
fscores_backgr = scores[(dvalue > cut) & (y != 0)]
fweight_signal = weight[(dvalue > cut) & (y == 0)]
fweight_backgr = weight[(dvalue > cut) & (y != 0)]

#Defining the binning
numbins = 16
b, c = np.histogram(fscores_backgr, weights=fweight_backgr, 
                    bins= numbins, range=(-1,1))
s, c = np.histogram(fscores_signal, weights=fweight_signal, 
                    bins=c)

#Compute the significance (root squared because sign_bin return 
#the squared value)
print(np.sqrt(sign_bin(s, b, 0.1, certain=False)))
print(np.sqrt(sign_bin(s, b, 0.05, certain=False)))
print(np.sqrt(sign_bin(s, b, 0.000000001, certain=True)))

#Plotting the BDT distribution
plotbin(y, scores, weight, dvalue, j = 0, binned = c, 
        ylim=1e9, cut = cut)

#If need for evaluating the other channel or model with different params,
#just copy the above code in this section or change the number in the function
