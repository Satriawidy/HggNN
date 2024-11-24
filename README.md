# HggNN
Repository for the project "Feasibility Study to Search for Higgs Decays to Gluons at the LHC Using Machine Learning"

The input data for running the codes are:
1. h5 file containing jet-level and track/calorimeter-level information of the Higgs jet candidates, used for training and testing the network
2. h5 files containing event-level information of the $VH\rightarrow gg$ candidates and track/calorimeter-level information of the corresponding Higgs jet candidate, used for significance analysis.

## Set Up
The code was tested on python 3.10.12. The requirements.txt file contains the required dependencies to run the code. It is advised to setup virtual environment for installing dependencies and running the code, although the original project did not do such things. Here we present the step to set up virtual environment and install dependencies in linux. This could also be done in other operating system, although the method might slightly differs.

### Setting Up Virtual Environment
Setting up virtual environment can be done using conda(here, we use 'hggnn' as the environment name), by running the following line on the terminal

```
conda create -n hggnn python=3.10.12
```

and everytime, the environment can be activated via

```
conda activate hggnn
```

### Installing Dependencies
After activating the environment, we can move on to installing the dependencies. This can be done either using pip or conda. To install using pip, run the following line on the terminal

```
python -m pip install -r requirements.txt
```
To install using conda, use the following line

```
conda install --yes --file requirements.txt
```

### Another Option
Another option is to install the dependencies while creating the environment in conda

```
conda create --name hggnn --file requirements.txt
```

## Processing the Graphs
Both track/calorimeter-level information of the Higgs jet candidates need to be processed into graphs before training and significance analysis. This is performed by uncommenting all commented lines in the `process.py` and `process_final.py` and run them by typing

```
python process.py
python process_final.py
```

in the terminal. Remember to comment the lines again after processing the graphs to prevent reprocessing the graphs. It normally took a few minutes to create the graphs, so a few lines to monitor the processing were added. Those lines should give update on how many graphs are processed every few (around six) minutes.

## Neural Network Training and Testing
### The Training
After the training graphs are created, they can be used to train and test the neural network. For the GNN, this is performed by modifying the parameter and running the script in `GNN_train.py`. To train a normal `ParticleNet` architecture, use the `NeuralNetGraph' class and tune the parameters wrapped here

```
model_R020 = NeuralNetGraph(
    module = ParticleNetMini([64, 128, 256, 512], 128, 0.7578, 16, 10, num_classes = len(weights)),
    criterion = nn.CrossEntropyLoss,
    optimizer = optim.Adam,
    criterion__reduction = 'none',
    criterion__weight = weights,
    verbose = 10,
    optimizer__lr = 0.04636,
    batch_size= 256,
    classes = [0,1,2,3,4],
    train_split=None,
    device = device,
    max_epochs = 11,
    iterator_train__num_workers = 16,
    iterator_valid__num_workers = 16,
    iterator_train__pin_memory = False,
    iterator_valid__pin_memory = False,
    callbacks = [reje_e, acce_e, stop, cp]
)
#Training
model_R020.fit(X_train, X_valid)
#Saving the model for evaluation use
model_R020.save_params(f_params='model_29th/model_R020.pkl')
```
Also do the same for training mass-decorrelated `ParticleNet`, but with `NeuralNetGraphDiscorr` class instead. More than one model can be initialised and trained in one file, but the training process will be performed after each other. Remember to change the models' name when training new models to make sure that the previous models are not erased. Finally, run the script by typing

```
python GNN_train.py
```

in the terminal

### The Testing
Testing the GNN is performed in several steps. First, initialise the model (or models) in `GNN_test.py` the same way as in the training procedure, with the exception of the last few lines

```
model_R020.initialize()  # This is important!
model_R020.load_params(f_params='model_29th/model_R020.pkl')
```

for each model. Then call the testing lines

```
models = [model_R020, model_M020]
model_name = ['R020', 'M020']
model_legend = ['GNN 20', 'GNN DisCo 20']

#Extracting discriminant and score using model_eval
#Discriminant and score is evaluated on test dataset
model_eval(models, model_name, model_legend, 'r')
```
according to the information about the models that we want to test. The `model_name` will appear in the filename, while the `model_legend` is usually used to describe the model in the plot legend. Running the script

```
python GNN_test.py
```

will produce h5 files containing the network output and discriminant. These files can then be processed to obtain further results and plots by modifying the following lines in `GNN_plots.py`

```
model_name = ['R020', 'M020']
model_legend = ['GNN 20', 'GNN DisCo 20']

evalplot(model_name, model_legend, 1)
```

according to which model we want to put together in the plots. For example, the above lines will produce results and individual plots for model `M020` and model `R020`, and combined plots of the two models, marked with number `1` in the plots' filename. After modifying the lines and saving the files, just run the script by typing

```
python GNN_plots.py
```

in the terminal.

## Significance Analysis
To perform significance analysis, we do the following procedures:
1. Produce GNN discriminant for each Higgs jet candidate using the optimised and mass-decorrelated `ParticleNet`.
2. Perform BDT training using the event-level information of the $VH\rightarrow gg$ candidates to obtain the BDT score distribution.
3. Apply cut on the events using the GNN discriminant and other event-level information.
4. Compute the binned significance analysis using the BDT score distribution of the surviving events.

### Producing GNN discriminant
To produce GNN discriminant using a certain model, simply initialise the model in `GNN_final.py` the same way as in the model testing, and put the model name in the following lines

```
#Evaluate the chosen model on evaluation (significance analysis) dataset
for filename in os.listdir("hfivesdir/0L/"):
    if filename not in os.listdir("resultdir/0L/"):
        print(filename)
        data = dataset(direname = '0L', filename = filename)
        eval_final(model_M020, data, '0L', filename)
for filename in os.listdir("hfivesdir/1L/"):
    if filename not in os.listdir("resultdir/1L/"):
        print(filename)
        data = dataset(direname = '1L', filename = filename)
        eval_final(model_M020, data, '1L', filename)
for filename in os.listdir("hfivesdir/2L/"):
    if filename not in os.listdir("resultdir/2L/"):
        print(filename)
        data = dataset(direname = '2L', filename = filename)
        eval_final(model_M020, data, '2L', filename)
```

followed by running the script by typing

```
GNN_final.py
```

in the terminal.

### Remaining Steps

The remaining steps are taken care in the `BDT.py` file. To train the BDT, initialise the BDT parameters and uncomment the last three lines in the following snapshot

```
#-------------------------Main Training------------------------------#
#Initialising the parameters for training BDT
params_1 = {'loss':'exponential', 'learning_rate':0.1, 
            'n_estimators':600, 'max_depth':8, 'verbose':10,
            'subsample':0.25}

##Uncomment these lines to train model A and model B for all channels with params_1
#train_bdt(0, params_1, 1)
#train_bdt(1, params_1, 1)
#train_bdt(2, params_1, 1)
```

while changing the params name according to the desired name. If we already have the trained BDT, we can proceed to the last two steps by keeping those three lines commented and instead modify the following line

```
#-------------------------Main Evaluation----------------------------#
#Examples for evaluating the BDT on 0-lepton channel with model using params_1
y, scores, weight, dvalue = test_bdt(0, 1)
```

depending on which BDT model we want to use and which lepton-channel we want to evaluate on. If we want to do more than one evaluation, we can also simply copy the lines and modify the BDT model and lepton-channel. Finally, both the BDT training and significance analysis are performed by typing

```
python BDT.py
```

in the terminal.
