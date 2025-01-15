using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs, discretize_distribution

k_distribution = discretize_distribution(Normal(10, 2.5), 4:20) # Was 2 initially but could not support epidemics

parameter_combinations = [
    (N=100, k=k_distribution, p=.0),
    (N=200, k=k_distribution, p=0.),
    (N=250, k=k_distribution, p=.0),
    (N=300, k=k_distribution, p=.0),
    (N=400, k=k_distribution, p=.0),
    (N=500, k=k_distribution, p=.0),
    (N=600, k=k_distribution, p=.0),
    (N=700, k=k_distribution, p=.0),
    (N=750, k=k_distribution, p=0.),
    (N=800, k=k_distribution, p=.0),
    (N=900, k=k_distribution, p=.0),
    (N=1000, k=k_distribution, p=.0),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(watts_strogatz, N_graphs, parameter_combinations, @__DIR__, "gridmd")
