#%%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd
import networkx as nx
import dataset as ds
import models
import matplotlib.pyplot as plt
#%%
# Create training dataset
DATA_PATH = "/home/DATA/datasets/SIS_teps_archive/"
networks_dir = os.path.join(DATA_PATH, "networks")
teps_dir = os.path.join(DATA_PATH, "teps")
train_dataset = ds.EdgePredictionDataset(
    networks_dir=networks_dir,
    teps_dir=teps_dir,
    network_type='erdos',
    num_train=10,
    num_test=1,
    is_training=True,
)

train_dataset = ds.balance_dataset(dataset=train_dataset)

# Create test dataset
test_dataset = ds.EdgePredictionDataset(
    networks_dir=networks_dir,
    teps_dir=teps_dir,
    network_type='erdos',
    num_train=5,
    num_test=1,
    is_training=False,
)

test_dataset = ds.balance_dataset(dataset=test_dataset)

# Create data loaders
from torch.utils.data import DataLoader
batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Example of iterating through the dataset
for batch_features, batch_labels in train_loader:
    print(f"Feature shape: {batch_features.shape}")
    print(f"Labels shape: {batch_labels.shape}")
    break

for batch_features, batch_labels in train_loader:
    print("Features stats:")
    print(f"Mean: {batch_features.mean():.4f}")
    print(f"Std: {batch_features.std():.4f}")
    print(f"Min: {batch_features.min():.4f}")
    print(f"Max: {batch_features.max():.4f}")
    print("\nLabels distribution:", torch.bincount(batch_labels.squeeze().long()))
    break

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 3) Instantiate model
#%%
model = models.PairLSTMClassifier(
    lstm_hidden_dim=64,
    mlp_hidden_dim=32
).to(device)

# Learning rate scheduling and optimization
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    weight_decay=0.1,
)


criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0]).to(device))
models.train_with_logging_and_cm(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, batch_size=batch_size, threshold=0.5, device=device)

# %%
