using Graphs
using Random
Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, k=5),
    (N=200, k=5),
    (N=250, k=5),
    (N=300, k=5),
    (N=400, k=5),
    (N=500, k=5),
    (N=600, k=5),
    (N=700, k=5),
    (N=750, k=5),
    (N=800, k=5),
    (N=900, k=5),
    (N=1000, k=5),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(barabasi_albert, N_graphs, parameter_combinations, @__DIR__, "ba")
