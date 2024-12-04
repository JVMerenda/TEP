using Graphs
using Random
using NPZ
Random.seed!(1234)

function generate(N_graphs, parameter_combinations)
    for params in parameter_combinations
        output_dir = "N$(params.N)"
        isdir(output_dir) || mkdir(output_dir)
        for i in 1:N_graphs
            g = random_regular_graph(params.N, params.k)
            npzwrite(joinpath(output_dir, "reg-$i.npz"), adjacency_matrix(g))
        end
    end
end

cd(@__DIR__)

parameter_combinations = [
    (N=100, k=10),
    (N=250, k=10),
    (N=500, k=10),
    (N=1000, k=10),
]

try
    N_graphs = parse(Int, ARGS[1])
    generate(N_graphs, parameter_combinations)
catch
    @error "Usage: julia build_graphs.jl N_graphs"
end
