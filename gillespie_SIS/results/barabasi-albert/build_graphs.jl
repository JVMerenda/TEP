using Graphs
using Random
using NPZ
Random.seed!(1234)

cd(@__DIR__)

parameter_combinations = [
    (N=100, k=5),
    (N=250, k=5),
    (N=500, k=5),
    (N=1000, k=5),
]
N_graphs = 10

for params in parameter_combinations
    output_dir = "N$(params.N)"
    isdir(output_dir) || mkdir(output_dir)
    for i in 1:N_graphs
        g = barabasi_albert(params.N, params.k, params.k)
        npzwrite(joinpath(output_dir, "ba-$i.npz"), adjacency_matrix(g))
    end
end
