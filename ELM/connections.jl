using Associations

"""
    create_words(sequence, word_length)

From a bitvector , create an array of integers of bit length `word_length` from the sequence of bits in the bitvector.
"""
function create_words(sequence, word_length)
    kernel = [2^i for i in 0:word_length-1]
    words = [sequence[i:i+word_length-1]' * kernel for i in 1:length(sequence)-word_length+1]
    return words
end

function mutual_information_matrix(tep; word_length=1)
    n_vertices = size(tep, 2)
    mutual_info = zeros(Float64, n_vertices, n_vertices)
    word_vectors = [create_words(row, word_length) for row in eachrow(tep)]
    est = JointProbabilities(MIShannon(), UniqueElements())
    for i in 1:n_vertices
        for j in i:n_vertices
            mutual_info[i,j] = association(est, word_vectors[i], word_vectors[j])
            mutual_info[j,i] = mutual_info[i,j]
        end
    end
    return mutual_info
end

function correlation_matrix(tep; word_length=1, cor=PearsonCorrelation)
    n_vertices = size(tep, 2)
    correlation = zeros(Float64, n_vertices, n_vertices)
    word_vectors = [create_words(row, word_length) for row in eachrow(tep)]
    for i in 1:n_vertices
        for j in i:n_vertices
            try
                correlation[i,j] = association(cor(), word_vectors[i], word_vectors[j])
            catch e
                if isa(e, DomainError)
                    correlation[i,j] = 0.0
                else
                    rethrow(e)
                end
            end

            correlation[j,i] = correlation[i,j]
        end
    end
    return correlation
end

function infection_count_matrix(tep)
    n_vertices = size(tep, 2)
    counts = zeros(Float64, n_vertices, n_vertices)
    for i in 1:n_vertices
        for t in 1:size(tep, 1)-1
            if tep[t,i] == 0 && tep[t+1,i] == 1
                for j in 1:n_vertices
                    counts[i,j] += tep[t,j]
                end
            end
        end
    end
    return .5*(counts .+ counts')
end
