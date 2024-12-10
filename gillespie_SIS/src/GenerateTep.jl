module GenerateTep

using ArgParse
using Distributions
using Graphs
using NPZ

__precompile__()

export SIS
export MSIS
export parse_command_line_args
export read_graph
export discretize_distribution
export geometric_graph
export build_graphs
export copy_graph

include(joinpath(@__DIR__, "gillespie_sis.jl"))
include(joinpath(@__DIR__, "gillespie_msis.jl"))
include(joinpath(@__DIR__, "preparation.jl"))

end
