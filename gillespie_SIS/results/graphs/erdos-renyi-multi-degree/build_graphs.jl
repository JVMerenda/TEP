using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, p=Beta(3, 27)),
    (N=250, p=Beta(1.5, 36)),
    (N=500, p=Beta(4, 96)),
    (N=1000, p=Beta(1.2,118.8)),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(erdos_renyi, N_graphs, parameter_combinations, @__DIR__, "ermd")
