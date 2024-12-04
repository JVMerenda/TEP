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

function generate(N_graphs, parameter_combinations)
    for params in parameter_combinations
        output_dir = "N$(params.N)"
        isdir(output_dir) || mkdir(output_dir)
        for i in 1:N_graphs
            g = geometric_graph(params.N, params.D, params.p)
            npzwrite(joinpath(output_dir, "geom-$i.npz"), adjacency_matrix(g))
        end
    end
end

cd(@__DIR__)

parameter_combinations = [
    (N=100, D=0.198, p=0.1),
    (N=250, D=0.119, p=0.1),
    (N=500, D=0.083, p=0.1),
    (N=1000, D=0.058, p=0.1),
]

try
    N_graphs = parse(Int, ARGS[1])
    generate(N_graphs, parameter_combinations)
catch
    @error "Usage: julia build_graphs.jl N_graphs"
end
