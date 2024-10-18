using NetworkJumpProcesses
using Graphs
using JumpProcesses
using ArgParse
using NPZ

"""
This script is meant to be used by command line, with optional arguments defined in the function below.

Generate a random graph using the Erdős-Rényi model, simulate stochastic SIS model on it and store a tep of the results.
For each invocation one single new graph is generated, and the SIS model is simulated on it `N` times.

The simulated process is exact in continuous time. By the optional argument --dt it is possible to sample the tep at discrete time steps.
If not given, the exact tep is stored as a list of time points and the vertex indices that change state at that point.

Example usage:
```bash
julia --project generate_tep.jl --n 900 --N 4 --dt 1.
```
"""

function parse_command_line_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n"
        help = "Number of vertices"
        arg_type = Int64
        default = 1000

        "--p"
        help = "Edge probability"
        arg_type = Float64
        default = 0.01

        "--N"
        help = "Number of teps for the same graph"
        arg_type = Int64
        default = 1

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

        "--dt"
        help = "Sampling step; if nothing is given, the exact tep is returned"
        arg_type = Float64
        default = nothing
    end

    return parse_args(s)
end

"""
    generate_jump_set(graph)

Generate the jump set for the SIS model on a network `graph`.
"""
function generate_jump_sets(graph)
    # Healing event (intra vertex jump)
    v_IS = ConstantJumpVertex(
        (v, nghbs, p, t) -> v[1]*p[2], # rate
        (v, nghbs, p, t) -> v[1] = 0 # affect!
    )

    # Infection event (jump over an edge)
    e_SI = ConstantJumpEdge(
        (vs, vd, p, t) -> vs[1]*p[1], # rate
        (vs, vd, p, t) -> vd[1] = 1 # affect!
    )

    return network_jump_set(graph; vertex_reactions=[v_IS], edge_reactions=[e_SI])
end

"""
    Construct the jump problem with a single random  infectious node and calculate a solution.
"""
function solve_problem(λ, μ, n, T, jset, vtj, jtv)
    p = (λ, μ)
    u₀ = zeros(Int64, n)
    u₀[rand(eachindex(u₀))] = 1
    dprob = DiscreteProblem(u₀, (0., T), p)
    rssa_jump_prob = JumpProblem(dprob, RSSA(), jset; vartojumps_map=vtj, jumptovars_map=jtv)
    return solve(rssa_jump_prob, SSAStepper())
end

"""
    to_tep(sol)

Convert the solution `sol` to an exact tep given by a two row matrix, where the first row
contains the time points and the second row contains the vertex indices.
"""
function to_tep(sol)
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
function to_tep(sol, dt)
    return hcat([sol(t) for t in 0:dt:sol.t[end]]...)
end

function main()
    args = parse_command_line_args()
    gg = erdos_renyi(args["n"], args["p"])
    jset = generate_jump_sets(gg)
    vtj = vartojumps(gg, 1, 1, 1)
    jtv = jumptovars(gg, 1, 1, 1)
    npzwrite("graph.npz", adjacency_matrix(gg))
    Threads.@threads for i in 1:args["N"]
        sol = solve_problem(args["lambda"], args["mu"], args["n"], args["T"], jset, vtj, jtv)
        tep = isnothing(args["dt"]) ? to_tep(sol) : to_tep(sol, args["dt"])
        npzwrite("tep-$i.npz", tep)
    end
    return 0
end

main()
