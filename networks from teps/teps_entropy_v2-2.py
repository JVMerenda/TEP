import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#This function imports the TEP matrix
def tep_matrix(directory,file):
    data = np.load(directory+file)
    return data

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

#ploting the word's distribution
def plot(word_vec, node_label):
    plt.hist(word_vec, density=True, bins=128)
    plt.xlabel('word')
    plt.ylabel('Frequency')
    plt.title('Frequency distribution of node '+str(node_label))
    plt.grid()
    plt.savefig('dist_node'+str(node_label)+'.png')
    plt.show()

def entropy(data):
    #Calculate the entropy of a list of decimal integers. (word decimal list)
    counts = Counter(data)
    total = len(data)
    return -sum((count / total) * np.log2(count / total)
                for count in counts.values() if count > 0)

#Create a list of Si entropies (each node entropy)
def entropies(TEP_matrix):
    N = len(TEP_matrix)
    S_vec = []
    for i in range(N):
      decimal_i = generate_10bit_sequences(TEP_matrix[i])
      S_i = entropy(decimal_i)
      S_vec.append(S_i)
    return S_vec


def joint_entropy(data1, data2):
    #Calculate the joint entropy between two nodes.
    assert len(data1) == len(data2), "Both lists must have the same length!"

    joint_pairs = list(zip(data1, data2))  #Connecting two lists on the same index
    joint_counts = Counter(joint_pairs)    #counting uniques words from the pair list
    total = len(joint_pairs)
    #Now, the joint entropy
    return -sum((count / total) * np.log2(count / total)
                for count in joint_counts.values() if count > 0)

def generate_10bit_sequences(arr):
    #Generate a list of decimal integers from 10-bit sequences.
    #Firt we generate a 10-bit word, then we convert it to decimals
    n = len(arr) - 10 + 1
    sequences = [arr[i:i + 10] for i in range(n)]  #sequence of 10-bit words
    decimal_values = [int("".join(map(str, seq)), 2) for seq in sequences]
    return decimal_values

#Creating the S matrix, where Sij terms represent the joint entropy from pair (i,j)
def joint_entropy_matrix(TEP_matrix):
    N = len(TEP_matrix)
    S_matrix = [[0 for _ in range(N)] for _ in range(N)]
    #Let's scan the TEP matrix and generates joint entropies from it
    for i in range(N):
        for j in range(N):
            decimal_i = generate_10bit_sequences(TEP_matrix[i])
            decimal_j = generate_10bit_sequences(TEP_matrix[j])
            Sij = joint_entropy(decimal_i,decimal_j)
            S_matrix[i][j] = Sij

    return S_matrix

#The delta Matrix, where Delta_ij = Si + Sj - Sij
#This Matrix represent the mutual relationship between two nodes
def delta_matrix(S_matrix, S_vec):
    N = len(S_vec)
    Delta = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            delta_ij = S_vec[i] + S_vec[j] - S_matrix[i][j]
            Delta[i][j] = delta_ij
    return Delta

#This Funcion shows the value distribution into the S_matrix and D_matrix
#You can comment it if you won't use it
def matrix_representation(S_matrix, D_matrix):
    plt.figure()
    plt.imshow(D_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Intensity')
    plt.savefig('D_matrix')

    plt.figure()
    plt.imshow(S_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Intensity')
    plt.savefig('S_matrix')



#Building the network from the  adjacency matrix
def building_net(A, file):
    file = file.replace('.npz', '')
    matrix = np.array(A)
    G = nx.from_numpy_array(matrix)
    N = len(G.nodes())
    nx.write_gml(G, "Graph_N"+str(N)+"_"+file+".gml")
    return G

#Plot network. You can comment it too
def plot_net(G):
    #To turn the visualizatiom easier, let's catch only the largest connected component
    connected_components = nx.connected_components(G)
    # Extract the largest connected component (with the most nodes)
    largest_cc = max(connected_components, key=len)
    # Create a subgraph containing only the nodes from the largest component
    G = G.subgraph(largest_cc).copy()
    plt.figure()
    pos = nx.spring_layout(G)  # Layout for a nice spread-out graph
    # Normalize weights to use them as alpha values (transparency)
    # Plot the graph with weights on the edges
    nx.draw_networkx_nodes(G, pos, node_color='orange', node_size=50)
    nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=2, edge_color='blue')
    plt.axis('off')
    plt.savefig('graph.png')


#Main function. here is the core of our code
def main(TEP_matrix):
    #S matrix, Delta matrix, S list, and its visualization
    S_matrix = joint_entropy_matrix(TEP_matrix)
    S_vec = entropies(TEP_matrix)
    D_matrix = delta_matrix(S_matrix, S_vec)
    
    #matrix_representation(S_matrix, D_matrix)  #Let it commented if you won't use it

    '''Let's choice a good threshold.
       I suggest to use k-means method choosing the best K'''
    D_matrix = np.array(D_matrix)
    D_flat = D_matrix[np.triu_indices_from(D_matrix, k=1)].reshape(-1, 1)
    #Silhouette method to choose the best k
    silhouette_scores = []
    K_range = range(2, 10)
    for K in K_range:
        kmeans = KMeans(n_clusters=K, random_state=42).fit(D_flat)
        score = silhouette_score(D_flat, kmeans.labels_)
        silhouette_scores.append(score)

    optimal_K = np.argmax(silhouette_scores) + 1  # Best k (based on silhouette Score)
    kmeans = KMeans(n_clusters=optimal_K, random_state=42).fit(D_flat)

    # Defining the threshold as the mean point between the two largest centroids
    centroids = np.sort(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centroids[-2:])  # mean of the two largest centroids

    # Applying the threshold to create the adjacency matrix
    A = (D_matrix > threshold).astype(int)
    np.fill_diagonal(A, 0)  # Remove self-loops


    #Making The graph
    G = building_net(A, file)
    #plot_net(G)


'''In this code, we do this procedure for only one TEP. But we can generalize it
put a 'for' loop over all TEPs file in directory below
'''
directory = '/home/DATA/datasets/SIS_teps/N100/'
file = 'tep-1-19-0.1.npz'
M = tep_matrix(directory, file)
main(M)
