using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs, geometric_graph

parameter_combinations = [
    (N=100, D=Exponential(0.198), p=0.1),
    (N=200, D=Exponential(0.134), p=0.1),
    (N=250, D=Exponential(0.119), p=0.1),
    (N=300, D=Exponential(0.108), p=0.1),
    (N=400, D=Exponential(0.093), p=0.1),
    (N=500, D=Exponential(0.083), p=0.1),
    (N=600, D=Exponential(0.076), p=0.1),
    (N=700, D=Exponential(0.070), p=0.1),
    (N=750, D=Exponential(0.067), p=0.1),
    (N=800, D=Exponential(0.065), p=0.1),
    (N=900, D=Exponential(0.061), p=0.1),
    (N=1000, D=Exponential(0.058), p=0.1),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(geometric_graph, N_graphs, parameter_combinations, @__DIR__, "geomd")
