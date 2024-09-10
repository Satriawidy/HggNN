# HggNN
Repository for the project "Feasibility Study to Search for Higgs Decays to Gluons at the LHC Using Machine Learning"

The input data for running the codes are:
1. h5 file containing jet-level and track/calorimeter-level information of the Higgs jet candidates, used for training and testing the network
2. h5 files containing event-level of the $VH\rightarrow gg$ candidates and track/calorimeter-level information of the corresponding Higgs jet candidate, used for significance analysis.

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
