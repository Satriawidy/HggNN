# HggNN
Repository for the project "Feasibility Study to Search for Higgs Decays to Gluons at the LHC Using Machine Learning"

The input data for running the codes are:
1. h5 file containing jet-level and track/calorimeter-level information of the Higgs jet candidates, used for training and testing the network
2. h5 files containing event-level of the $VH\rightarrow gg$ candidates and track/calorimeter-level information of the corresponding Higgs jet candidate, used for significance analysis.

## Processing the Graphs
Both track/calorimeter-level information of the Higgs jet candidates need to be processed into graphs before training and significance analysis. This is performed by uncommenting all commented lines in the `process.py` and `process_final.py` and run them

```
python process.py
python process_final.py
```

Remember to comment them again after processing the graphs to prevent reprocessing the graphs again and again. It normally took a few minutes to create the graphs, so a few lines to monitor the processing were added. Those lines should give update on how many graphs are processed every few (around six) minutes.

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
Also do the same for training mass-decorrelated `ParticleNet`, but with `NeuralNetGraphDiscorr` class instead. Also remember to change the model name to make sure that the previous models are not erased.

### The Testing
Testing the GNN is performed in several steps. First, initialise the model in `GNN_test.py` the same way as in training
