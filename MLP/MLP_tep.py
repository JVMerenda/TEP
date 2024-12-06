import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from mutual_information import mutual_information_matrix
from read_tep import SIS_TEP

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, 64)          
        self.fc3 = nn.Linear(64, 1)           
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def train_and_test(train_data, train_labels, test_data, test_labels, epochs=100, learning_rate=0.001):
    train_data = np.atleast_2d(train_data).T if train_data.ndim == 1 else train_data
    test_data = np.atleast_2d(test_data).T if test_data.ndim == 1 else test_data
    input_size = train_data.shape[1]
    model = MLP(input_size)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32).view(-1, 1)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
        # Update the learning rate
        scheduler.step()

    model.eval()
    with torch.no_grad():
        predictions = model(test_data)
        print(predictions)
        predictions = (predictions >= 0.5).float().squeeze()  # Convertendo para classes binárias
        accuracy = accuracy_score(test_labels, predictions)
        print(f"Acurácia no conjunto de teste: {accuracy:.4f}")
        predicted_classes = predictions.tolist()
        return predicted_classes


#Well, let's build the frequence distribution
#It calculates the frequence distribution for a single nodes
def distribution(word_vec):
    total_count = len(word_vec)
    freq_counter = Counter(word_vec)
    probabilities = {num: freq / total_count for num, freq in freq_counter.items()}
    #ploting the distribution
    x = probabilities.keys()
    y = probabilities.values()
    return x, y

def building_net(A):
    return nx.from_numpy_array(A)

def infer_p_edges(G):
    n = G.number_of_nodes()
    E = G.number_of_edges()
    p = (2 * E) / (n * (n - 1))  # Probabilidade de ligação
    return p

'''In this code, we do this procedure for only one TEP. But we can generalize it
put a 'for' loop over all TEPs file in directory below
'''
import os
graph_model = "er" # choose from er, ba, ws, geo, euc, sf, reg, grid
graph_size = 100 # choose from 100, 250, 500, 1000
dt =.1

i_graph = 1 # choose from 1, .., 50
j_tep = 1 # choose from 1, .., 100

exact_tep = SIS_TEP(graph_model, graph_size, i_graph, j_tep)
M_train = exact_tep.sample(0.1)
M_test = exact_tep.sample(1.)

z = exact_tep.load_graph()

mutual_info_matrices = []
for i_graph in tqdm(range(1, 51)):
    for j_tep in tqdm(range(1, 101)):
        exact_tep = SIS_TEP(graph_model, graph_size, i_graph, j_tep)
        mutual_info_matrices.append(exact_tep.load_or_generate_mutual_info(dt))


y = z.flatten()
X_train = mutual_information_matrix(M_train)
X_test = mutual_information_matrix(M_test)




