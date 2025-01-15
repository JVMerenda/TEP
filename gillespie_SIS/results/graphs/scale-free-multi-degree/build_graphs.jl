using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs, discretize_distribution

parameter_combinations = [
    (N=100, m=discretize_distribution(Normal(500, 50), 250:750), α=2.5),
    (N=200, m=discretize_distribution(Normal(1000, 100), 500:1500), α=2.5),
    (N=250, m=discretize_distribution(Normal(1250, 125), 750:1750), α=2.5),
    (N=300, m=discretize_distribution(Normal(1500, 150), 1000:2000), α=2.5),
    (N=400, m=discretize_distribution(Normal(2000, 200), 1500:2500), α=2.5),
    (N=500, m=discretize_distribution(Normal(2500, 250), 1500:3500), α=2.5),
    (N=600, m=discretize_distribution(Normal(3000, 300), 2000:4000), α=2.5),
    (N=700, m=discretize_distribution(Normal(3500, 350), 2500:4500), α=2.5),
    (N=750, m=discretize_distribution(Normal(3750, 375), 2500:5000), α=2.5),
    (N=800, m=discretize_distribution(Normal(4000, 400), 3000:5000), α=2.5),
    (N=900, m=discretize_distribution(Normal(4500, 450), 3000:6000), α=2.5),
    (N=1000, m=discretize_distribution(Normal(5000, 500), 2500:7500), α=2.5),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(static_scale_free, N_graphs, parameter_combinations, @__DIR__, "sfmd")
