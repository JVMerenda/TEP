# %%
import os
import numpy as np
import pandas as pd
import math
import torch
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.utils import shuffle

# Number of networks to load
number_of_networks = 10
sample_number = number_of_networks

# Load the train and test dataset
train_data = pd.read_csv('/home/DATA/datasets/TEP/MLP/train_dataset.csv', nrows=number_of_networks, header=None).values  # M x N
train_labels = pd.read_csv('/home/DATA/datasets/TEP/MLP/train_labels.csv', nrows=number_of_networks, header=None).values  # M x N

test_data = pd.read_csv('/home/DATA/datasets/TEP/MLP/train_dataset.csv', nrows=number_of_networks+1, header=None).values  # c x N
test_labels = pd.read_csv('/home/DATA/datasets/TEP/MLP/train_labels.csv', nrows=number_of_networks+1, header=None).values  # c x N

# %%
train_samples = []
train_targets = []
squared_dim = int(math.sqrt(train_data.shape[1]))
pair_indices = torch.triu_indices(squared_dim, squared_dim, offset=1)

for n in range(number_of_networks):
    train_sample = train_data[n].reshape(squared_dim, squared_dim)
    train_sample = np.vstack([np.concatenate((train_sample[:, i], train_sample[:, j]))
                               for i, j in zip(pair_indices[0], pair_indices[1])])
    train_samples.append(train_sample)

    train_target = train_labels[n].reshape(squared_dim, squared_dim)
    train_target = np.array([train_target[i, j] for i, j in zip(pair_indices[0], pair_indices[1])])
    train_targets.append(train_target)

train_samples = np.vstack(train_samples)  # Convert to numpy array
train_targets = np.hstack(train_targets)  # Convert to numpy array

# Balance the dataset
positive_indices = np.where(train_targets == 1)[0]
negative_indices = np.where(train_targets == 0)[0]

num_positive = len(positive_indices)
num_negative = len(negative_indices)
num_samples = min(num_positive, num_negative)

# Randomly sample from the majority class to balance the dataset
balanced_positive_indices = np.random.choice(positive_indices, num_samples, replace=False)
balanced_negative_indices = np.random.choice(negative_indices, num_samples, replace=False)

# Combine and shuffle the balanced dataset
balanced_indices = np.concatenate((balanced_positive_indices, balanced_negative_indices))
np.random.shuffle(balanced_indices)

train_samples = train_samples[balanced_indices]
train_targets = train_targets[balanced_indices]

# Prepare test sample
test_sample = test_data[sample_number].reshape(squared_dim, squared_dim)
test_sample = np.vstack([np.concatenate((test_sample[:, i], test_sample[:, j]))
                         for i, j in zip(pair_indices[0], pair_indices[1])])
test_label = test_labels[sample_number].reshape(squared_dim, squared_dim)
test_label = np.array([test_label[i, j] for i, j in zip(pair_indices[0], pair_indices[1])])

# %%
# Train Linear SVM
print("Training Linear SVM...")
svm_model = SVC(kernel="linear", probability=True, random_state=42)
svm_model.fit(train_samples, train_targets)

# %%
# Make predictions on the test set
test_predictions = svm_model.predict(test_sample)

# Compute evaluation metrics
overall_accuracy = np.mean(test_predictions == test_label)
precision = precision_score(test_label, test_predictions, zero_division=0)
recall = recall_score(test_label, test_predictions, zero_division=0)

print(f"\nAverage accuracy of test set: {overall_accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")

# Generate and display confusion matrix
cm = confusion_matrix(test_label, test_predictions, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - SVM Test Set")
plt.savefig("confusion_matrix_svm.jpg")
plt.show()

# %%
