#%%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
import matplotlib.pyplot as plt
import math
from torch.optim.lr_scheduler import CosineAnnealingLR

# Path to save the model weights
model_path = "mlp_weights.pth"

number_of_networks = 100
sample_number = number_of_networks

# Load the train and test dataset
train_data = pd.read_csv('/home/DATA/datasets/TEP/MLP/train_dataset.csv', nrows=number_of_networks, header=None).values  # M x N
train_labels = pd.read_csv('/home/DATA/datasets/TEP/MLP/train_labels.csv', nrows=number_of_networks, header=None).values  # M x N

test_data = pd.read_csv('/home/DATA/datasets/TEP/MLP/train_dataset.csv', nrows=number_of_networks+1, header=None).values  # c x N
test_labels = pd.read_csv('/home/DATA/datasets/TEP/MLP/train_labels.csv', nrows=number_of_networks+1, header=None).values  # c x N

#%%
train_samples = torch.empty(0)
train_targets = torch.empty(0)
squared_dim = int(math.sqrt(train_data.shape[1]))
pair_indices = torch.triu_indices(squared_dim, squared_dim, offset=1)

for n in range(number_of_networks):
    train_sample = torch.tensor(train_data[n].reshape(squared_dim, squared_dim))
    train_sample = torch.vstack([torch.cat((train_sample[:,i], train_sample[:,j]))
                    for i, j in zip(pair_indices[0], pair_indices[1])])
    train_samples = torch.cat((train_samples, train_sample))    

    train_target = train_labels[n].reshape(squared_dim, squared_dim)
    train_target = torch.tensor([train_target[i,j]
                                for i, j in zip(pair_indices[0], pair_indices[1])
                            ])
    train_targets = torch.cat((train_targets, train_target))

# Balance the dataset
positive_indices = (train_targets == 1).nonzero(as_tuple=True)[0]
negative_indices = (train_targets == 0).nonzero(as_tuple=True)[0]

# Determine the number of samples to match the minority class
num_positive = len(positive_indices)
num_negative = len(negative_indices)
num_samples = min(num_positive, num_negative)

# Randomly sample from the majority class to balance the dataset
balanced_positive_indices = positive_indices[torch.randperm(num_positive)[:num_samples]]
balanced_negative_indices = negative_indices[torch.randperm(num_negative)[:num_samples]]

# Combine the indices and shuffle them
balanced_indices = torch.cat((balanced_positive_indices, balanced_negative_indices))
balanced_indices = balanced_indices[torch.randperm(len(balanced_indices))]

# Select the balanced samples and targets
train_samples = train_samples[balanced_indices]
train_targets = train_targets[balanced_indices]

test_sample = torch.tensor(test_data[sample_number].reshape(squared_dim, squared_dim))
test_sample = torch.vstack([torch.cat((test_sample[:,i], test_sample[:,j]))
                   for i, j in zip(pair_indices[0], pair_indices[1])])
test_label = test_labels[sample_number].reshape(squared_dim, squared_dim)
test_label = torch.tensor([test_label[i,j]
    for i, j in zip(pair_indices[0], pair_indices[1])
])

    
# Dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# make the dataloader
train_dataset = CustomDataset(train_samples, train_targets)
test_dataset = CustomDataset(test_sample, test_label)


batch_size = 8192
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# MLP architecture
# MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MLP, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # Add Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),  # Add Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.BatchNorm1d(hidden_size // 8),  # Add Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 8, hidden_size // 32),
            nn.BatchNorm1d(hidden_size // 32),  # Add Batch Normalization
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 32, output_size),
            nn.Sigmoid()  # Binary output
        ).to(device)
    
    def forward(self, x):
        x = x.to(device)
        return self.model(x)


# Model Settings
input_size = train_samples.shape[1]  # size 2*N
hidden_size = 4096
output_size = 1  # output 1

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP(input_size, hidden_size, output_size, device).to(device)
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
num_epochs = 1000
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0)

# Inicializar lista para salvar erros por época
epoch_losses = []

# # Verify if the model already was trained 
# if os.path.exists(model_path):
#     # Load the saved weights
#     model.load_state_dict(torch.load(model_path))
#     print("Pesos do modelo carregados com sucesso.")
# else:
    # Train
print("Training The Model...")
epoch_losses = []
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device)).squeeze()
        loss = criterion(outputs, targets.to(device).float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Step the scheduler at the end of each epoch
    scheduler.step()

    epoch_losses.append(epoch_loss / len(train_loader))
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

# Save the weights
torch.save(model.state_dict(), model_path)
print(f"Treinamento concluído.")

# Plot the evolution of the loss
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), epoch_losses, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss per epoch")
plt.legend()
plt.grid(True)
plt.savefig('erro.jpg')

# Evaluation of the test set
model.eval()
all_targets = []
all_outputs = []

with torch.no_grad():
    accuracies = []
    for i, (inputs, targets) in enumerate(test_loader):
        outputs = model(inputs).round().cpu()  # Binary output (0 or 1)

        # Calculate accuracy for the batch
        correct = (outputs == targets).sum().item()
        accuracy = correct / targets.size(0)  # Accuracy
        accuracies.append(accuracy)

        # Flatten targets and outputs to accumulate for global metrics
        all_targets.extend(targets.numpy().flatten())
        all_outputs.extend(outputs.numpy().flatten())

# Calculate overall accuracy
overall_accuracy = np.mean(accuracies)
print(f"\nAverage accuracy of test set: {overall_accuracy:.2%}")

# Calculate precision and recall
precision = precision_score(all_targets, all_outputs, zero_division=0)
recall = recall_score(all_targets, all_outputs, zero_division=0)
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")

# Generate and display the overall confusion matrix
cm = confusion_matrix(all_targets, all_outputs, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Overall Test Set")
plt.savefig("confusion_matrix_overall.jpg")
plt.show()
# %%
