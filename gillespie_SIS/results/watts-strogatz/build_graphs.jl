using Graphs
using Random
using NPZ
Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, k=10, p=.1),
    (N=250, k=10, p=.1),
    (N=500, k=10, p=.1),
    (N=1000, k=10, p=.1),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(watts_strogatz, N_graphs, parameter_combinations, @__DIR__, "ws")
