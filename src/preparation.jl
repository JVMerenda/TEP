Base.Vector{Float64}(s::String) = Float64.([parse(Float64, item) for item in split(s[2:end-1], ",")[1:end-1]])

"""
    parse_command_line_args()

"""
    parse_command_line_args()

function parse_command_line_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--N_graphs"
        help = "Number of graphs to generate"
        arg_type = Int64
        default = 1
        
        "--N_vertices"
        help = "Number of vertices per graph"
        arg_type = Int64
        default = 1000

        "--N_teps"
        help = "Number of teps per graph"
        arg_type = Int64
        default = 1

        "--p"
        help = "Edge probability"
        arg_type = Float64
        default = 0.01

        "--lambda"
        help = "Infection rate"
        arg_type = Float64
        default = 0.03

        "--mu"
        help = "Healing rate"
        arg_type = Float64
        default = 0.09

        "--T"
        help = "Time period"
        arg_type = Float64
        default = 100.0

        "--output"
        help = "Output directory"
        arg_type = String
        default = "."

        "--dt"
        help = "Sampling step; if nothing is given, the exact tep is returned"
        arg_type = Vector{Float64}
        default = nothing
    end

    return parse_args(s)
end

"""
    to_tep(sol)

Convert the solution `sol` to an exact tep given by a two row matrix, where the first row
contains the time points and the second row contains the vertex indices.
"""
function to_tep(sol::ODESolution)
    t = Vector{Float64}(undef, length(sol.t))
    x = Vector{Int64}(undef, length(sol.t))
    t[1] = 0.
    x[1] = findfirst(sol.u[1] .== 1)[1]
    c_idx = 1
    for i in 2:length(t)
        options = findall(sol.u[i] .!= sol.u[i-1])
        if length(options) > 0
            c_idx += 1
            t[c_idx] = sol.t[i]
            x[c_idx] = options[1]
        elseif length(options) > 1
            @error "Multiple changes at the same time"
        end
    end
    return [t[1:c_idx] Float64.(x[1:c_idx])]
end

"""
    to_tep(sol, dt)

Convert the solution `sol` to a sampled tep in the form of a matrix.
Each column corresponds to a time point and each row corresponds to a vertex.
"""
function to_tep(sol::ODESolution, dt::Real)
    return hcat([sol(t) for t in 0:dt:sol.t[end]]...)
end
