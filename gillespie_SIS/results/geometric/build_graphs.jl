using Graphs
using Random
using NPZ
Random.seed!(1234)

function geometric_graph(N, D, p; d=2)
    g = euclidean_graph(N, d; cutoff=D)[1]
    for e in edges(g)
        if rand() < p
            fixed = rand([e.src, e.dst])
            rem_edge!(g, e)
            new = rand(setdiff(1:N, [fixed]))
            add_edge!(g, fixed, new)
        end
    end
    return g
end

cd(@__DIR__)

parameter_combinations = [
    (N=100, D=0.198, p=0.1),
    (N=500, D=0.083, p=0.1),
    (N=1000, D=0.058, p=0.1),
]
N_graphs = 10

for params in parameter_combinations
    output_dir = "N$(params.N)"
    isdir(output_dir) || mkdir(output_dir)
    for i in 1:N_graphs
        g = geometric_graph(params.N, params.D, params.p)
        npzwrite(joinpath(output_dir, "graph-$i.npz"), adjacency_matrix(g))
    end
end
