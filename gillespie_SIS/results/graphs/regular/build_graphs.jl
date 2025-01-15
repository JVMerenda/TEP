using Graphs
using Random
Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, k=10),
    (N=200, k=10),
    (N=250, k=10),
    (N=300, k=10),
    (N=400, k=10),
    (N=500, k=10),
    (N=600, k=10),
    (N=700, k=10),
    (N=750, k=10),
    (N=800, k=10),
    (N=900, k=10),
    (N=1000, k=10),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(random_regular_graph, N_graphs, parameter_combinations, @__DIR__, "reg")
