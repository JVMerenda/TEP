const UNIT_INTERVAL_BINS = range(0, 1, length=21)

"""
    create_words(sequence, word_length)

From a bitvector, create an array of integers of bit length `word_length` from the sequence of bits in the bitvector.
"""
function create_words(sequence::AbstractArray{<:Integer}, word_length)
    kernel = [2^i for i in 0:word_length-1]
    words = [sequence[i:i+word_length-1]' * kernel for i in 1:length(sequence)-word_length+1]
    return words
end

function create_words(sequence::AbstractArray{<:Real}, word_length)
    return sequence
end

function get_probabilities(x::AbstractArray{<:Integer})
    return probabilities(x)
end

function get_probabilities(x::AbstractArray{<:Real})
    vb = ValueBinning(FixedRectangularBinning((UNIT_INTERVAL_BINS, )))
    return probabilities_and_outcomes(vb, x)[1]
end

function binning(x::AbstractArray{<:Integer})
    return UniqueElements()
end

function binning(x::AbstractArray{<:Real})
    return CodifyVariables(ValueBinning(FixedRectangularBinning((UNIT_INTERVAL_BINS, ))))
end

"""
    mutual_information_matrix(tep; word_length=1)

Compute the mutual information matrix of a time-evolving probability matrix `tep` using words of length `word_length`.
"""
function mutual_information_matrix(tep::AbstractArray; word_length=1)
    n_vertices = size(tep, 2)
    mutual_info = Matrix{Float64}(undef, n_vertices, n_vertices)
    word_vectors = [create_words(row, word_length) for row in eachcol(tep)]

    for i in 1:n_vertices
        mutual_info[i, i] = information(Shannon(), get_probabilities(word_vectors[i]))
    end

    est_joint = JointProbabilities(JointEntropyShannon(), binning(word_vectors[1]))
    for i in 1:n_vertices
        for j in i+1:n_vertices
            joint_entropy = association(est_joint, word_vectors[i], word_vectors[j])
            mutual_info[i, j] = mutual_info[i, i] + mutual_info[j, j] - joint_entropy
        end
    end

    return Symmetric(mutual_info)
end
