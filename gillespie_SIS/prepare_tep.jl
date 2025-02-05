using NPZ
using ProgressMeter
include(joinpath(@__DIR__, "src", "GenerateTep.jl"))
using .GenerateTep

network_size = 100
dt = 0.1

train_graph_idxs = 1:40
test_graph_idxs = 41:50
train_tep_idxs = 1:20
test_tep_idxs = 21:25

input_dir = joinpath(@__DIR__, "results/", "sis")
output_dir = joinpath(@__DIR__, "..", "data", "teps", "N$(network_size)")

graph_models = ["ba", "er", "euc", "geo", "reg", "sf", "ws"]
graph_models = vcat(graph_models, graph_models .* "md")
push!(graph_models, "gridmd")

train_dir = joinpath(output_dir, "train")
test_dir = joinpath(output_dir, "test")
ispath(train_dir) || mkpath(train_dir)
ispath(test_dir) || mkpath(test_dir)

pb = Progress(length(graph_models) * (length(train_graph_idxs) + length(test_graph_idxs)))
for graph_model in graph_models
    @Threads.threads for graph_i in train_graph_idxs ∪ test_graph_idxs
        graph = UInt8[]
        graph_name = ""
        train_teps = Matrix{UInt8}[]
        test_teps = Matrix{UInt8}[]
        for tep_j in train_tep_idxs ∪ test_tep_idxs
            tep = TEP(graph_model, network_size, graph_i, tep_j)
            if isempty(graph)
                graph = UInt8.(vectorize_upper_triangular(load_graph(tep); include_diagonal=false))
                graph_name = "$(graph_model)-$(tep.location.graph_id)"
            end
            sampled_tep = UInt8.(sample_with_dt(tep, dt))
            if graph_i in test_graph_idxs || tep_j in test_tep_idxs
                push!(test_teps, sampled_tep)
            else
                push!(train_teps, sampled_tep)
            end
        end
        !isempty(train_teps) ? npzwrite(joinpath(train_dir, "$(graph_name).npz"), graph=graph, train_teps...) : nothing
        !isempty(test_teps) ? npzwrite(joinpath(test_dir, "$(graph_name).npz"), graph=graph, test_teps...) : nothing
        next!(pb)
    end
end
