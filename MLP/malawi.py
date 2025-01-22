import read_datasets as rd
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data  
        self.labels = labels  

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], output_size),
            nn.Sigmoid()  # Binary output
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, criterion, optimizer, device, num_epochs=150, class_weights=torch.tensor([0.5, 0.5])):
    print("Training the model...")
    epoch_losses = []
    class_weights = class_weights.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)  # [batch_size, 4950, 200] and [batch_size, 4950]

            # Process each input separately
            for sample_input, sample_target in train_loader:
                optimizer.zero_grad()

                # Forward pass
                outputs = model(sample_input).squeeze()  
                loss_per_sample = criterion(outputs, sample_target)
                weights = class_weights[targets.long().squeeze()]
                loss = (loss_per_sample * weights).mean()

                # Backward pass
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        epoch_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
    
    return epoch_losses

# Plot the evolution of the loss
def plot_loss(epoch_losses, save_path='err_malawi_full.jpg'):
    plt.figure(figsize=(8, 6))
    num_epochs = len(epoch_losses)
    plt.plot(range(1, num_epochs + 1), epoch_losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss per epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

def evaluate_model(model, test_loader, device='cpu', threshold=.5):
    model.eval()
    model.to(device)
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, target in test_loader:
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(target.cpu().numpy().flatten().round())
            
    predictions = [1 if p > threshold else 0 for p in predictions]
    return predictions, [int(t) for t in targets]

## Shared parameters
batch_size = 16
class_weights = torch.tensor([0.1, 0.9])
device = 0

hidden_sizes = [64, ]
output_size = 1
criterion = nn.BCELoss()  # Binary Cross-Entropy
lr = 0.001
num_epochs = 150


## FULL  PERIOD
data, labels = rd.handle_dataset(rd.BASE_DIR + "graphs/real/malawi/full")
data = data[0, :, :].to(device)
labels = labels[0, :].to(device)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=123)

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = data.shape[1]  # size 2*N
class_weights = torch.tensor([y_train.mean(), 1. - y_train.mean()])
fp_model = MLP(input_size, hidden_sizes, output_size).to(device)
optimizer = torch.optim.Adam(fp_model.parameters(), lr=lr)

fp_epoch_losses = train_model(fp_model, train_loader, criterion, optimizer, device, num_epochs=num_epochs, class_weights=class_weights)
plot_loss(fp_epoch_losses, save_path='err_malawi_full.jpg')
fp_predictions, fp_targets = evaluate_model(fp_model, test_loader, device)

fp_acc = accuracy_score(fp_targets, fp_predictions)
fp_prc = precision_score(fp_targets, fp_predictions)
fp_rcl = recall_score(fp_targets, fp_predictions)
fp_cm = confusion_matrix(fp_targets, fp_predictions)


## PER WEEK
train_data, train_labels = rd.handle_dataset(rd.BASE_DIR + "graphs/real/malawi/week_01")
test_data, test_labels = rd.handle_dataset(rd.BASE_DIR + "graphs/real/malawi/week_02")
train_data, test_data = train_data[0, :, :].to(device), test_data[0, :, :]
train_labels, test_labels = train_labels[0, :].to(device), test_labels[0, :]

train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = train_data.shape[1]  # size 2*N
class_weights = torch.tensor([y_train.mean(), 1. - y_train.mean()])
week_model = MLP(input_size, hidden_sizes, output_size).to(device)
optimizer = torch.optim.Adam(week_model.parameters(), lr=lr)

week_epoch_losses = train_model(week_model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
plot_loss(week_epoch_losses, save_path='err_malawi_week.jpg')
week_predictions, week_targets = evaluate_model(week_model, test_loader, device)

week_acc = accuracy_score(week_targets, week_predictions)
week_prc = precision_score(week_targets, week_predictions)
week_rcl = recall_score(week_targets, week_predictions)
week_cm = confusion_matrix(week_targets, week_predictions)


## PER DAY
test_days = [4, 13]
all_days = range(1, 15)
train_files = [f"day_{day:02d}" for day in all_days if day not in test_days]
test_files = [f"day_{day:02d}" for day in test_days]
train_data, train_labels = rd.handle_multiple_datasets(rd.BASE_DIR + "graphs/real/malawi", train_files)
train_data, train_labels = train_data.to(device), train_labels.to(device)
test_data, test_labels = rd.handle_multiple_datasets(rd.BASE_DIR + "graphs/real/malawi", test_files)

train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = train_data.shape[2]  # size 2*N
class_weights = torch.tensor([y_train.mean(), 1. - y_train.mean()])
day_model = MLP(input_size, hidden_sizes, output_size).to(device)
optimizer = torch.optim.Adam(day_model.parameters(), lr=lr)

day_epoch_losses = train_model(day_model, train_loader, criterion, optimizer, device, num_epochs=num_epochs)
plot_loss(day_epoch_losses, save_path='err_malawi_day.jpg')
day_predictions, day_targets = evaluate_model(day_model, test_loader, device)
