using Graphs
using Random
using Distributions

Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, p=Beta(3, 27)),
    (N=200, p=Beta(1.5, 28.5)),
    (N=250, p=Beta(2, 48)),
    (N=300, p=Beta(2, 58)),
    (N=400, p=Beta(2, 78)),
    (N=500, p=Beta(2, 98)),
    (N=600, p=Beta(2, 118)),
    (N=700, p=Beta(2, 138)),
    (N=750, p=Beta(2, 148)),
    (N=800, p=Beta(2, 158)),
    (N=900, p=Beta(2, 178)),
    (N=1000, p=Beta(2,198)),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(erdos_renyi, N_graphs, parameter_combinations, @__DIR__, "ermd")
