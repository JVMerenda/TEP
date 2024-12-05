import numpy as np

def joint_entropy(data1, data2):
    """
    Calcula a entropia conjunta entre dois conjuntos de dados de forma otimizada.
    """
    # Combine data1 e data2 em uma única matriz com 2 colunas
    joint_pairs = np.stack((data1, data2), axis=-1)

    # Usa uma abordagem vetorizada para contar combinações únicas
    unique, counts = np.unique(joint_pairs, axis=0, return_counts=True)
    probabilities = counts / len(data1)  # Probabilidade de cada par
    return -np.sum(probabilities * np.log2(probabilities))

def joint_entropy_matrix(TEP_matrix, sequences):
    """
    Calcula a matriz de entropia conjunta (S_matrix) de forma otimizada.
    """
    N = len(TEP_matrix)
    S_matrix = np.zeros((N, N))

    # Calcula apenas a metade superior da matriz, aproveitando a simetria
    for i in range(N):
        for j in range(i, N):
            # Calcula a entropia conjunta
            S_matrix[i, j] = joint_entropy(sequences[i], sequences[j])
            if i != j:  # Simetria
                S_matrix[j, i] = S_matrix[i, j]

    return S_matrix


# Gerar sequências de 10 bits e convertê-las para decimais
def generate_10bit_sequences(arr):
    n = len(arr) - 10 + 1
    return [int("".join(map(str, arr[i:i + 10])), 2) for i in range(n)]

# Cálculo da entropia
def entropy(data):
    unique, counts = np.unique(data, return_counts=True)
    probabilities = counts / len(data)
    return -np.sum(probabilities * np.log2(probabilities))


# Calcular a matriz Delta
def delta_matrix(S_matrix, S_vec):
    N = len(S_matrix)
    Delta = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            delta_ij = S_vec[i] + S_vec[j] - S_matrix[i, j]
            Delta[i, j] = delta_ij
            Delta[j, i] = delta_ij  # Simetria
    return Delta


def mutual_information_matrix(TEP_matrix):
    """Função principal para calcular D."""
    sequences = [generate_10bit_sequences(row) for row in TEP_matrix]
    S_vec = [entropy(sequences[i]) for i in range(len(sequences))]
    S_matrix = joint_entropy_matrix(TEP_matrix, sequences)
    D_matrix = delta_matrix(S_matrix, S_vec)
    return D_matrix.flatten()
