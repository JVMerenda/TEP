using Graphs
using Random
Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, p=.1),
    (N=250, p=.04),
    (N=500, p=.02),
    (N=1000, p=.01),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(erdos_renyi, N_graphs, parameter_combinations, @__DIR__, "er")
