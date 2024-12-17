import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Path to save the model weights
model_path = "mlp_weights.pth"

# Load the train and test dataset
train_data = pd.read_csv('train_dataset.csv', header=None).values  # M x N
train_labels = pd.read_csv('train_labels.csv', header=None).values  # M x N

test_data = pd.read_csv('test_dataset.csv', header=None).values  # c x N
test_labels = pd.read_csv('test_labels.csv', header=None).values  # c x N

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
train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)  

# MLP architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # Binary output
        )
    
    def forward(self, x):
        return self.model(x)

# Model Settings
input_size = train_data.shape[1]  # size N
hidden_size = 4096
output_size = train_labels.shape[1]  # Size N too

model = MLP(input_size, hidden_size, output_size)
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Inicializar lista para salvar erros por época
epoch_losses = []

# Verify if the model already was trainned 
if os.path.exists(model_path):
    # Load the saved weights
    model.load_state_dict(torch.load(model_path))
    print("Pesos do modelo carregados com sucesso.")
else:
    # Train
    print("Treining The Model...")
    num_epochs = 200
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_losses.append(epoch_loss / len(train_loader))
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss / len(train_loader):.4f}")
    
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
with torch.no_grad():
    accuracies = []
    for i, (inputs, targets) in enumerate(test_loader):
        outputs = model(inputs).round()  # Binary output (0 or 1)
        correct = (outputs == targets).sum(dim=1).item()
        accuracy = correct / targets.size(1)  # Accuracy
        accuracies.append(accuracy)
        print(f"Test {i+1}: Accuracy = {accuracy:.2%}")
        
        # Confusion matrices
        targets_flat = targets.numpy().flatten()
        outputs_flat = outputs.numpy().flatten()
        cm = confusion_matrix(targets_flat, outputs_flat, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion matrix {i+1}")
        plt.savefig(f"confusion_matrix_{i+1}.jpg")

    overall_accuracy = np.mean(accuracies)
    print(f"\nAverage accuracy of test set: {overall_accuracy:.2%}")