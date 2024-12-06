module SIS

using Graphs
using NetworkJumpProcesses
using JumpProcesses

const StateType = UInt8
const susceptible = zero(StateType)
const infectious = one(StateType)
"""
    generate_jump_set(graph)

Generate the jump set for the SIS model on a network `graph`.
"""
function generate_jump_sets(graph)
    # Healing event (intra vertex jump)
    v_IS = ConstantJumpVertex(
        (v, nghbs, p, t) -> v[1]*p[2], # rate
        (v, nghbs, p, t) -> v[1] = susceptible  # affect!
    )

    # Infection event (jump over an edge)
    e_SI = ConstantJumpEdge(
        (vs, vd, p, t) -> vs[1]*(1. - vd[1])*p[1], # rate
        (vs, vd, p, t) -> vd[1] = infectious  # affect!
    )

    jump_set = network_jump_set(graph; vertex_reactions=[v_IS], edge_reactions=[e_SI])
    vtj = vartojumps(graph, 1, 1, 1)
    jtv = jumptovars(graph, 1, 1, 1)

    return jump_set, vtj, jtv
end

"""
    Construct the jump problem with a single random  infectious node and calculate a solution.
"""
function solve_problem(λ, μ, n, T, jset, vtj, jtv; δ=nothing, ppn=nothing)
    p = (λ, μ)
    u₀ = zeros(StateType, n)
    u₀[rand(eachindex(u₀))] = infectious
    dprob = DiscreteProblem(u₀, (0., T), p)
    rssa_jump_prob = JumpProblem(dprob, RSSA(), jset; vartojumps_map=vtj, jumptovars_map=jtv)
    return solve(rssa_jump_prob, SSAStepper())
end

function is_success(sol::ODESolution, allow_dieout::Bool)
    return allow_dieout || any(u[1] == 1 for u in sol[end])
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

function plot_density(sol::ODESolution, ts, graph, g_name, j)
    densities = [count(sol(t) .== 1) / nv(graph) for t in ts]
    return plot(ts, densities, title="Graph $(g_name), TEP $j"; legend=false)
end

end
