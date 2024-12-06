import numpy as np
from collections import Counter

def generate_words(X, word_size=10):
    '''
    Generates binary words of a specified size from the input array.

    Parameters:
    X (list or array-like): Input binary sequence.
    word_size (int): Length of each word to generate (default is 10).

    Returns:
    list: A list of integers, where each integer represents a binary word of length `word_size`.

    Explanation:
    The function slides a window of size `word_size` over the input sequence `X` and concatenates the binary digits
    within the window into a single integer. The integers are represented in base 2.
    '''
    n = len(X) - word_size + 1
    return np.array([int("".join(map(str, X[i:i + word_size])), 2) for i in range(n)],dtype=np.int32)


def entropy(X):
    '''
    Calculates the entropy of a given discrete distribution.

    Parameters:
    X (list or array-like): Input sequence of discrete values.

    Returns:
    float: The entropy of the input sequence, measured in bits.

    Explanation:
    The function uses the frequency counts of unique elements in `X` to compute the entropy based on the formula:
        H(X) = -Σ(p * log2(p))
    where `p` represents the relative frequency of each unique value in the sequence.

    Note:
    A `TODO` comment indicates a possible performance bottleneck due to slow casting of counts.
    '''
    counts = np.array(tuple(Counter(X).values()))  # Slow casting noted in TODO.
    total = np.sum(counts)
    return -sum((counts / total) * np.log2(counts / total))


def joint_entropy(X, Y):
    '''
    Calculates the joint entropy of two discrete distributions.

    Parameters:
    X (list or array-like): First input sequence.
    Y (list or array-like): Second input sequence.

    Returns:
    float: The joint entropy of the two input sequences.

    Explanation:
    The function computes the joint entropy by treating each pair `(X[i], Y[i])` as a single combined symbol
    and applying the entropy formula to the resulting joint distribution.

    Note:
    Similar to `entropy`, this function has a potential performance bottleneck noted in the TODO comment.
    '''
    counts = np.array(tuple(Counter(zip(X, Y)).values()))  # Slow casting noted in TODO.
    total = np.sum(counts)
    return -sum((counts / total) * np.log2(counts / total))


def mutual_entropy(X, Y):
    '''
    Calculates the mutual information between two discrete distributions.

    Parameters:
    X (list or array-like): First input sequence.
    Y (list or array-like): Second input sequence.

    Returns:
    float: The mutual information between `X` and `Y`, measured in bits.

    Explanation:
    Mutual information is computed as:
        I(X; Y) = H(X) + H(Y) - H(X, Y)
    where `H(X)` is the entropy of `X`, `H(Y)` is the entropy of `Y`, and `H(X, Y)` is their joint entropy.
    '''
    return entropy(X) + entropy(Y) - joint_entropy(X, Y)


def mutual_information_matrix(TEP_matrix, word_size=10):
    '''
    Constructs a mutual information matrix for a given data matrix.

    Parameters:
    TEP_matrix (numpy.ndarray): 2D array where each column represents a variable and each row represents an observation.
    word_size (int): Length of the binary words to generate for each variable (default is 10).

    Returns:
    numpy.ndarray: A square matrix where the element at (i, j) represents the mutual information between
                   the `i`-th and `j`-th variables.

    Explanation:
    The function calculates mutual information for all pairs of columns in `TEP_matrix` by first generating binary
    words for each column and then applying `mutual_entropy` on each pair of columns.

    Note:
    A `TODO` comment highlights a potential inefficiency caused by the nested loop structure.
    '''
    T, N = TEP_matrix.shape
    TEP_word = np.zeros((T - word_size + 1, N), dtype=np.int32)
    D_matrix = np.zeros((N, N), dtype=np.float32)
    for k in range(N):
        TEP_word[:, k] = generate_words(TEP_matrix[:, k], word_size=word_size)
    #====
    # alternative 1-line but slightly less efficient
    # TEP_word = np.array( [generate_words(TEP_matrix[:,k],word_size=word_size) for k in range(N)],dtype=np.int32).transpose()
    #=====
    # Compute mutual information for all pairs of variables
    for i in range(N):
        for j in range(i, N):
            D_matrix[i, j] = mutual_entropy(TEP_word[:, i], TEP_word[:, j])
            D_matrix[j, i] = D_matrix[i, j]  # Symmetric matrix.

    return D_matrix.flatten()






## NOTE this implementation is too slow. The version below scales poorly with TEP_matrix sizes

# def joint_entropy(data1, data2):
#     """
#     Calcula a entropia conjunta entre dois conjuntos de dados de forma otimizada.
#     """
#     # Combine data1 e data2 em uma única matriz com 2 colunas
#     joint_pairs = np.stack((data1, data2), axis=-1)

#     # Usa uma abordagem vetorizada para contar combinações únicas
#     unique, counts = np.unique(joint_pairs, axis=0, return_counts=True)
#     probabilities = counts / len(data1)  # Probabilidade de cada par
#     return -np.sum(probabilities * np.log2(probabilities))

# def joint_entropy_matrix(TEP_matrix, sequences):
#     """
#     Calcula a matriz de entropia conjunta (S_matrix) de forma otimizada.
#     """
#     N = len(TEP_matrix)
#     S_matrix = np.zeros((N, N))

#     # Calcula apenas a metade superior da matriz, aproveitando a simetria
#     for i in range(N):
#         for j in range(i, N):
#             # Calcula a entropia conjunta
#             S_matrix[i, j] = joint_entropy(sequences[i], sequences[j])
#             if i != j:  # Simetria
#                 S_matrix[j, i] = S_matrix[i, j]

#     return S_matrix


# # Gerar sequências de 10 bits e convertê-las para decimais
# def generate_10bit_sequences(arr):
#     n = len(arr) - 10 + 1
#     return [int("".join(map(str, arr[i:i + 10])), 2) for i in range(n)]


    
# Cálculo da entropia
# def entropy(data):
#     unique, counts = np.unique(data, return_counts=True)
#     probabilities = counts / len(data)
#     return -np.sum(probabilities * np.log2(probabilities))



# # Calcular a matriz Delta
# def delta_matrix(S_matrix, S_vec):
#     N = len(S_matrix)
#     Delta = np.zeros((N, N))
#     for i in range(N):
#         for j in range(i, N):
#             delta_ij = S_vec[i] + S_vec[j] - S_matrix[i, j]
#             Delta[i, j] = delta_ij
#             Delta[j, i] = delta_ij  # Simetria
#     return Delta


# def mutual_information_matrix(TEP_matrix):
#     """Função principal para calcular D."""
#     sequences = [generate_10bit_sequences(row) for row in TEP_matrix]
#     S_vec = [entropy(sequences[i]) for i in range(len(sequences))]
#     S_matrix = joint_entropy_matrix(TEP_matrix, sequences)
#     D_matrix = delta_matrix(S_matrix, S_vec)
#     return D_matrix.flatten()
