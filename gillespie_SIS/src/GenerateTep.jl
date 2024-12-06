module GenerateTep

using ArgParse
using Graphs
using NetworkJumpProcesses
using JumpProcesses
using NPZ

__precompile__()

export SIS
export MSIS
export parse_command_line_args
export read_graph
export geometric_graph
export build_graphs

include(joinpath(@__DIR__, "gillespie_sis.jl"))
include(joinpath(@__DIR__, "gillespie_msis.jl"))
include(joinpath(@__DIR__, "preparation.jl"))

end
