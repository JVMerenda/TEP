using Graphs
using Random
Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, p=0.1),
    (N=200, p=0.05),
    (N=250, p=0.04),
    (N=300, p=0.0333),
    (N=400, p=0.025),
    (N=500, p=0.02),
    (N=600, p=0.0167),
    (N=700, p=0.0143),
    (N=750, p=0.0133),
    (N=800, p=0.0125),
    (N=900, p=0.0111),
    (N=1000, p=0.01),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(erdos_renyi, N_graphs, parameter_combinations, @__DIR__, "er")
