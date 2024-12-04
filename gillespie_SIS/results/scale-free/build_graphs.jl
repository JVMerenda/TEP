using Graphs
using Random
using NPZ
Random.seed!(1234)

cd(@__DIR__)

parameter_combinations = [
    (N=100, m=500, α=-2.5),
    (N=250, m=1250, α=-2.5),
    (N=500, m=2500, α=-2.5),
    (N=1000, m=5000, α=-2.5),
]
N_graphs = 10

for params in parameter_combinations
    output_dir = "N$(params.N)"
    isdir(output_dir) || mkdir(output_dir)
    for i in 1:N_graphs
        g = static_scale_free(params.N, params.m, params.α)
        npzwrite(joinpath(output_dir, "sf-$i.npz"), adjacency_matrix(g))
    end
end
