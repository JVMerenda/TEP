using Statistics
using LinearAlgebra
using ProgressMeter
using Random
using Revise

include(joinpath(@__DIR__, "connections.jl"))
include(joinpath(@__DIR__, "read_tep.jl"))
using .SIS_TEP

function sigmoid(x)
    return 1 ./ (1 .+ exp.(-x))
end

struct ELM{T<:Real}
    input_weights::Matrix{T}
    biases::Matrix{T}
    output_weights::AbstractArray{T}
end

"""
    ELM(X, y, n_hidden)

Fit an Extreme Learning Machine to the data `X` with target `y` and `n_hidden` hidden units.
"""
function ELM(X::AbstractArray{T}, y::AbstractArray{T}, n_hidden::Int) where T<:Real
    n_samples, n_features = size(X)
    input_weights = randn(T, n_features, n_hidden)
    biases = randn(T, 1, n_hidden)
    G = X * input_weights .+ biases
    hidden_layer = sigmoid.(G)
    output_weights = pinv(hidden_layer) * y

    return ELM{T}(input_weights, biases, output_weights)
end

"""
    ELM(X)

Predict the output of an ELM model on the data `X`.
"""
function (elm::ELM)(X::Array{T}) where T<:Real
    G = X * elm.input_weights .+ elm.biases
    hidden_layer = sigmoid.(G)
    return hidden_layer * elm.output_weights
end

struct COMBO_ELM{T<:Real}
    connections::Dict{String,  ELM{T}}
    combination::ELM{T}
end

"""
    input_to_combination(Xs)

Transform the outputs of the individual ELMs into a single input for the combination ELM.
"""
function input_to_combination(Xs)
    Xs = [X'[:] for X in Xs]
    return hcat(Xs...)
end

"""
    unpack_combination_output(y, n)

Unpack the output of the combination ELM into the vectors of the upper triangular part
of the adjacency matrix.

# Arguments
- `y::AbstractVector`: The output of the combination ELM.
- `n::Int`: The number of elements in the upper triangular part of the adjacency matrix.
"""
function unpack_combination_output(y, n)
    n_samples = Int(length(y) / n)
    return reshape(y, (n, n_samples))'
end

"""
    COMBO_ELM(connection_matrices, Y, n_hidden)

Train a combined ELM model on the matrices of connections `connection_matrices` with target `Y` and `n_hidden` hidden units.
"""
function COMBO_ELM(connection_matrices::Dict{String, <:Matrix{T}}, Y::Matrix{T}, n_hidden) where T <: Real
    elms = Dict{String, ELM{T}}()
    Ys = Matrix{Float64}[]
    for (name, X) in connection_matrices
        elm = ELM(X, Y, n_hidden)
        elms[name] = elm
        push!(Ys, elm(X))
    end
    X_comb = input_to_combination(Ys)
    Y_comb = Y'[:]
    comb_elm = ELM(X_comb, Y_comb, n_hidden)
    return COMBO_ELM(elms, comb_elm)
end

"""
    (c_elm::COMBO_ELM)(X)

Predict the output of a COMBO_ELM model on the data `X`.
"""
function(c_elm::COMBO_ELM{T})(X::Dict{String, <:Matrix{T}}) where T<:Real
    @assert keys(X) == keys(c_elm.connections)
    ys = [elm(X[k]) for (k, elm) in c_elm.connections]
    x_comb = input_to_combination(ys)
    y_comb = c_elm.combination(x_comb)
    return unpack_combination_output(y_comb, size(ys[1], 2))
end

"""
    vectorize_upper_triangle(mat)

Vectorize the upper triangular part of a matrix.
"""
function vectorize_upper_triangle(mat::AbstractMatrix)
    n = size(mat, 1)
    v = zeros(Int(n*(n-1)//2))
    c_row = 1
    for i in 1:n
        for j in i+1:n
            v[c_row] = mat[i, j]
            c_row += 1
        end
    end
    return v
end

"""
    vector_to_adjacency(vec)

Transform a vectorized upper triangular part of a matrix into a symmetric matrix with zero diagonal.
"""
function vector_to_adjacency(vec::AbstractVector)
    n = Int(sqrt(2*length(vec) + 0.25) - 0.5)
    mat = zeros(Float64, n, n)
    c_row = 1
    for i in 1:n
        for j in i+1:n
            mat[i, j] = vec[c_row]
            mat[j, i] = vec[c_row]
            c_row += 1
        end
    end
    return mat
end

"""
    build_X_mats(mats)

Build a matrix of features of the upper triagular part from a vector of matrices.
"""
function build_X_mats(mats::Vector{<: Matrix{<: Real}})
    n_vertices = size(mats[1], 1)
    n_features = Int(n_vertices * (n_vertices - 1) // 2)
    n_samples = length(mats)
    X = Matrix{Float64}(undef, n_samples, n_features)

    for i in 1:n_samples
        X[i, :] .= vectorize_upper_triangle(mats[i])
    end

    return X
end

"""
    split_test_train(X, y, prop_train=.8)

Split the data `X` and `y` into training and testing sets. These can be either vectors or matrices.
"""
function split_test_train(X::AbstractArray, y::AbstractArray, prop_train=.8)
    n = size(X, 1)
    idx = shuffle(1:n)
    n_train = Int(n * prop_train)
    return X[idx[1:n_train], :], y[idx[1:n_train], :], X[idx[n_train+1:end], :], y[idx[n_train+1:end], :]
end


function load_teps(nb_vertices, network_types, graph_range, tep_range, dt)
    teps = Matrix{Float64}[]
    gs = Matrix{Float64}[]
    pb = Progress(length(network_types) * length(graph_range) * length(tep_range))
    for network_type in network_types
        for i_graph in graph_range
            for j_tep in tep_range
                tep = SIS_TEP.TEP(network_type, nb_vertices, i_graph, j_tep)
                push!(teps, sample_with_dt(tep, dt))
                push!(gs, load_graph(tep))
                next!(pb)
            end
        end
    end
    return teps, gs
end

"""
    build_connections(teps, connections)

Build the matrices of connections from the TEPs using the functions in the dictionary `connections`.
Returns a dictionary (with the same keys as `connections`) of matrices of connections.
"""
function build_connections(teps, connections)
    pb = Progress(length(teps))
    matrices = Dict{String, Vector{Matrix{Float64}}}()
    for (name, _) in connections
        matrices[name] = Matrix{Float64}[]
    end
    Threads.@threads for tep in teps
        for (name, f) in connections
            push!(matrices[name], f(tep))
        end
        next!(pb)
    end
    Xs = Dict{String, Matrix{Float64}}()
    for (name, mats) in matrices
        Xs[name] = build_X_mats(mats)
    end
    return Xs
end

connections = Dict([
    "mim1" => x -> mutual_information_matrix(x; word_length=1),
    "mim3" => x -> mutual_information_matrix(x; word_length=3),
    "mim5" => x -> mutual_information_matrix(x; word_length=5),
    "pc1" => x -> correlation_matrix(x; word_length=1),
#    "pc3" => x -> correlation_matrix(x; word_length=3),
    "dc1" => x -> correlation_matrix(x; word_length=1, cor=DistanceCorrelation),
#    "dc3" => x -> correlation_matrix(x; word_length=3, cor=DistanceCorrelation),
    "ic" => x -> infection_count_matrix(x)
])

teps, gs = load_teps(100, ["er"], 1:25, 1:20, 0.25)

train_teps, train_gs, test_teps, test_gs = vec.(split_test_train(teps, gs))

Y = build_X_mats(train_gs)
X_train = build_connections(train_teps, connections)
c_elm = COMBO_ELM(X_train, Y, 500)
c_elm(X_train)

Y_ref = build_X_mats(test_gs)
X_test = build_connections(test_teps, connections)

Y_pred = c_elm(X_test)

sqrt(mean((Y_pred .- Y_ref).^2))

As = [vector_to_adjacency(r) for r in eachrow(Y_pred .> .22)]
mean.(As)
Y_ref
Y_pred
