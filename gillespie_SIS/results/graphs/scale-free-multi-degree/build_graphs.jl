using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs, discretize_distribution

parameter_combinations = [
    (N=100, m=discretize_distribution(Normal(500, 50), 250:750), α=2.5),
    (N=250, m=discretize_distribution(Normal(1250, 125), 750:1750), α=2.5),
    (N=500, m=discretize_distribution(Normal(2500, 250), 1500:3500), α=2.5),
    (N=1000, m=discretize_distribution(Normal(5000, 500), 2500:7500), α=2.5),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(static_scale_free, N_graphs, parameter_combinations, @__DIR__, "sfmd")
