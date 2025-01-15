"""
    create_words(sequence, word_length)

From a bitvector, create an array of integers of bit length `word_length` from the sequence of bits in the bitvector.
"""
function create_words(sequence, word_length)
    kernel = [2^i for i in 0:word_length-1]
    words = [sequence[i:i+word_length-1]' * kernel for i in 1:length(sequence)-word_length+1]
    return words
end

"""
    mutual_information_matrix(tep; word_length=1)

Compute the mutual information matrix of a time-evolving probability matrix `tep` using words of length `word_length`.
"""
function mutual_information_matrix(tep; word_length=1)
    n_vertices = size(tep, 2)
    mutual_info = Matrix{Float64}(undef, n_vertices, n_vertices)
    word_vectors = [create_words(row, word_length) for row in eachcol(tep)]

    for i in 1:n_vertices
        mutual_info[i, i] = information(probabilities(word_vectors[i]))
    end

    est_joint = JointProbabilities(JointEntropyShannon(), UniqueElements())
    for i in 1:n_vertices
        for j in i+1:n_vertices
            joint_entropy = association(est_joint, word_vectors[i], word_vectors[j])
            mutual_info[i, j] = mutual_info[i, i] + mutual_info[j, j] - joint_entropy
        end
    end

    return Symmetric(mutual_info)
end
