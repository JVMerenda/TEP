module GenerateTep

using ArgParse
using ComplexityMeasures
using Associations
using Distributions
using Graphs
using Printf
using LinearAlgebra
using NPZ
using Statistics

__precompile__()

export SIS
export MSIS
export MMCA

export parse_command_line_args
export read_graph
export discretize_distribution
export geometric_graph
export build_graphs
export copy_graph
export vectorize_upper_triangular
export mutual_information_matrix
export mutual_info_from_tep

export TEP
export load_tep
export abrv_to_full
export get_tep_location
export get_graph_location
export get_mutual_info_location
export get_sample_location
export load_graph
export sample
export sample_with_dt
export sample_at_ts
export store_sample
export store_sample_p

include(joinpath(@__DIR__, "read_tep.jl"))
include(joinpath(@__DIR__, "gillespie_sis.jl"))
include(joinpath(@__DIR__, "gillespie_msis.jl"))
include(joinpath(@__DIR__, "mmca_sis.jl"))
include(joinpath(@__DIR__, "preparation.jl"))
include(joinpath(@__DIR__, "mutual_info.jl"))

end
