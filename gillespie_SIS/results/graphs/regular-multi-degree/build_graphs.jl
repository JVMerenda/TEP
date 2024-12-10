using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs, discretize_distribution

k_distribution = discretize_distribution(Normal(10, 2.5), 2:20)

parameter_combinations = [
    (N=100, k=k_distribution),
    (N=250, k=k_distribution),
    (N=500, k=k_distribution),
    (N=1000, k=k_distribution),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(random_regular_graph, N_graphs, parameter_combinations, @__DIR__, "regmd")
