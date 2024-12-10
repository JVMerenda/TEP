using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs, discretize_distribution

mean_degree_distribution = discretize_distribution(Gamma(5 / 1.5, 1.5), 2:12)

parameter_combinations = [
    (N=100, k=mean_degree_distribution),
    (N=250, k=mean_degree_distribution),
    (N=500, k=mean_degree_distribution),
    (N=1000, k=mean_degree_distribution),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(barabasi_albert, N_graphs, parameter_combinations, @__DIR__, "bamd")
