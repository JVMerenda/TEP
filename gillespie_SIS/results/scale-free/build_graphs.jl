using Graphs
using Random
using NPZ
Random.seed!(1234)

function generate(N_graphs, parameter_combinations)
    for params in parameter_combinations
        output_dir = "N$(params.N)"
        isdir(output_dir) || mkdir(output_dir)
        for i in 1:N_graphs
            g = static_scale_free(params.N, params.m, params.α)
            npzwrite(joinpath(output_dir, "sf-$i.npz"), adjacency_matrix(g))
        end
    end
end

cd(@__DIR__)

parameter_combinations = [
    (N=100, m=500, α=2.5),
    (N=250, m=1250, α=2.5),
    (N=500, m=2500, α=2.5),
    (N=1000, m=5000, α=2.5),
]

try
    N_graphs = parse(Int, ARGS[1])
    generate(N_graphs, parameter_combinations)
catch
    @error "Usage: julia build_graphs.jl N_graphs"
end
