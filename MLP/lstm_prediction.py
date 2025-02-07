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
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import models
import itertools
import dataset


# %%

network_class = "ba"
train_folder = "/home/DATA/datasets/TEP/data/teps/N100/train/"
test_folder = "/home/DATA/datasets/TEP/data/teps/N100/test/"
train_count = 4
test_count = 1

train_files = [os.path.join(train_folder, f"{network_class}-{i:02d}.npz") for i in range(1, train_count+1)]
test_files = [os.path.join(test_folder, f"{network_class}-{i:02d}.npz") for i in range(train_count+1, train_count+test_count+1)]
# Build dataset for training with window_size=100, stride=50
scaler = dataset.fit_scaler_on_train(train_files, file_count=20)

# 2) Create train_dataset from scaled data
train_dataset = dataset.create_full_dataset_with_scaling(
    file_paths=train_files,
    scaler=scaler,
    train=True,
    window_size=10,
    stride=10
)

# Build dataset for testing
test_dataset = dataset.create_full_dataset_with_scaling(
    file_paths=test_files,
    scaler=scaler,  # same scaler
    train=False,
    window_size=10,
    stride=10
)


# 2) Create DataLoaders
batch_size = 256

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 3) Instantiate model
model = models.PairLSTMClassifier(lstm_hidden_dim=512, mlp_hidden_dim=256).to(device)
#%%
# 4) Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

models.train_with_logging_and_cm(model, train_dataset, test_dataset, criterion, optimizer, num_epochs=100, batch_size=batch_size, device=device)
# %%
