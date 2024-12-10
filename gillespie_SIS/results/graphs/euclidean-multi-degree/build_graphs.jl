using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__,"..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs, geometric_graph

parameter_combinations = [
    (N=100, D=Exponential(0.198), p=0.0),
    (N=250, D=Exponential(0.119), p=0.0),
    (N=500, D=Exponential(0.083), p=0.0),
    (N=1000, D=Exponential(0.058), p=0.0),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(geometric_graph, N_graphs, parameter_combinations, @__DIR__, "eucmd")
