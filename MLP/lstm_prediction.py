#%%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
import models
import itertools
import dataset


# %%

network_class = "ba"
train_folder = "/home/DATA/datasets/TEP/data/teps/N100/train/"
test_folder = "/home/DATA/datasets/TEP/data/teps/N100/test/"
train_count = 1
test_count = 1

train_files = [os.path.join(train_folder, f"{network_class}-{i:02d}.npz") for i in range(1, train_count+1)]
test_files = [os.path.join(test_folder, f"{network_class}-{i:02d}.npz") for i in range(train_count+1, train_count+test_count+1)]

# Build dataset for training
train_dataset = dataset.create_full_dataset(train_files, train=True)

# Build dataset for testing
test_dataset  = dataset.create_full_dataset(test_files, train=False)
#%%

# 2) Create DataLoaders
batch_size = 16384
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# 3) Instantiate model
model = models.PairLSTMClassifier(lstm_hidden_dim=16, mlp_hidden_dim=32)

# 4) Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

models.train_with_logging(model, train_loader, test_loader, criterion, optimizer, num_epochs=100)
# %%
