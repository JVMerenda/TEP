using NPZ
include(joinpath(@__DIR__, "src", "GenerateTep.jl"))
using .GenerateTep

network_size = 100
dt = 0.1

train_graphs = 1:40
test_graphs = 41:50
train_teps = 1:20
test_teps = 21:25

input_dir = "results/sis/"
output_dir = "../data/sis/teps/N$(network_size)"

graph_models = ["er", "reg"]

train_data = Dict{String, Pair{Vector{UInt8}, Vector{Matrix{UInt8}}}}()
test_data = Dict{String, Pair{Vector{UInt8}, Vector{Matrix{UInt8}}}}()

for graph_model in graph_models
    for graph_i in train_graphs ∪ test_graphs
        graph = UInt8[]
        graph_name = ""
        train_teps = Matrix{UInt8}[]
        test_teps = Matrix{UInt8}[]
        for tep_j in train_teps ∪ test_teps
            tep = TEP(graph_model, network_size, graph_i, tep_j; nb_digits_g=1, nb_digits_tep=2)
            if isempty(graph)
                graph = UInt8.(vectorize_upper_triangular(load_graph(tep); include_diagonal=false))
                graph_name = "$(graph_model)-$(tep.location.graph_id)"
            end
            sampled_tep = UInt8.(sample_with_dt(tep, dt))
            if graph_i in test_graphs || tep_j in test_teps
                push!(test_teps, sampled_tep)
            else
                push!(train_teps, sampled_tep)
            end
        end
        !isempty(train_teps) ? train_data[graph_name] = train_teps : nothing
        !isempty(test_teps) ? test_data[graph_name] = test_teps : nothing
    end
end

train_dir = joinpath(output_dir, "train")
test_dir = joinpath(output_dir, "test")
ispath(train_dir) || mkpath(train_dir)
ispath(test_dir) || mkpath(test_dir)

for (file, (graph, teps)) in train_teps
    npzwrite(joinpath(train_dir, "$(file).npz"), graph=graph, teps...)
end

for (file, (graph, teps)) in test_teps
    npzwrite(joinpath(train_dir, "$(file).npz"), graph=graph, teps...)
end