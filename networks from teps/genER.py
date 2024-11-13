import networkx as nx
import random
import os
import pandas as pd

random.seed(42)

# Define the probability intervals
intervals = [
    (0.01, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.06),
    (0.06, 0.07), (0.07, 0.08), (0.08, 0.09), (0.09, 0.1), (0.1, 0.2),
    (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7),
    (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)
]

# Save the nets in this folder
directory = '/home/DATA/datasets/SIS_teps/networks/train_nets/'

#Function to create Erdos-Renyi networks
def create_and_save(int_idx, net_idx, p, n):
    G = nx.erdos_renyi_graph(n, p)
    file_name = f'{directory}{int_idx:02}_{net_idx:03}.gml'
    nx.write_gml(G, file_name)



# Loop to create nets to each probability interval
for int_idx, (p_min, p_max) in enumerate(intervals, start=1):
    for net_idx in range(1, 201):
        p = random.uniform(p_min, p_max)  # choose p in interval
        n = random.randint(100, 500)      # choose the number of nodes
        create_and_save(int_idx, net_idx, p, n)


#creating a CSV file representing the classes
classes = []
for i in range(1, 21):
    classe = str(i)  
    classes.extend([classe] * 200)  

df = pd.DataFrame(classes, columns=['Classe'])
df.to_csv(directory+'train_classes.csv', index=False)

print("Successful!")