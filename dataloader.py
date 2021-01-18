import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader 

# read csv file and make dataloader (a batch is single list of tensors)
def get_loader(args, name='noname', shuffle=True):
    data = TensorDataset(torch.from_numpy(np.loadtxt(os.path.join(args.path, args.data_dir, name), delimiter=',', dtype=np.float32)))
    return DataLoader(data, batch_size=args.batch_size, shuffle=shuffle)
    