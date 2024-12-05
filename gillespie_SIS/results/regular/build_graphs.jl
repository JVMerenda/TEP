using Graphs
using Random
Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, k=10),
    (N=250, k=10),
    (N=500, k=10),
    (N=1000, k=10),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(random_regular_graph, N_graphs, parameter_combinations, @__DIR__, "reg")
