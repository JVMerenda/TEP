using Graphs
using Random
using NPZ
Random.seed!(1234)

include(joinpath(@__DIR__, "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep: build_graphs

parameter_combinations = [
    (N=100, m=500, α=2.5),
    (N=200, m=1000, α=2.5),
    (N=250, m=1250, α=2.5),
    (N=300, m=1500, α=2.5),
    (N=400, m=2000, α=2.5),
    (N=500, m=2500, α=2.5),
    (N=600, m=3000, α=2.5),
    (N=700, m=3500, α=2.5),
    (N=750, m=3750, α=2.5),
    (N=800, m=4000, α=2.5),
    (N=900, m=4500, α=2.5),
    (N=1000, m=5000, α=2.5),
]

@assert length(ARGS) >= 1
N_graphs = parse(Int, ARGS[1])
build_graphs(static_scale_free, N_graphs, parameter_combinations, @__DIR__, "sf")
