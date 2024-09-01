import torch

torch.multiprocessing.set_sharing_strategy('file_system')

from evalplot import evalplot

model_name = ['R020', 'M020']
model_legend = ['GNN 20', 'GNN DisCo 20']

evalplot(model_name, model_legend, 1)