module MSIS

using Graphs
using NetworkJumpProcesses
using JumpProcesses
using Plots

"""
    generate_jump_set(graph)

Generate the jump set for the SIS model on a network `graph`.
"""
function generate_jump_sets(graph)
    # Infection event (intra vertex jump)
    v_SI = ConstantJumpVertex(
        (v, nghbs, p, t) -> p[1]*v[1]*v[2] / (v[1] + v[2] + 1e-6), # rate
        (v, nghbs, p, t) -> begin
            v[1] -= 1
            v[2] += 1
        end# affect!
    )

    # Healing event (intra vertex)
    v_IS = ConstantJumpVertex(
        (v, nghbs, p, t) -> p[2]*v[2], # rate
        (v, nghbs, p, t) -> begin
            v[1] += 1
            v[2] -= 1
        end# affect!
    )

    # Movement
    e_move = idx -> begin
        ConstantJumpEdge(
            (vs, vd, p, t) -> p[3]*vs[idx], # rate
            (vs, vd, p, t) -> begin
                vs[idx] -= 1
                vd[idx] += 1
            end
        )
    end
    e_mS = e_move(1)
    e_mI = e_move(2)

    jump_set = network_jump_set(graph; vertex_reactions=[v_IS, v_SI], edge_reactions=[e_mS, e_mI], nb_states=2)
    vtj = vartojumps(graph, 2, 2, 2)
    jtv = jumptovars(graph, 2, 2, 2)

    return jump_set, vtj, jtv
end

"""
    Construct the jump problem with a single random  infectious node and calculate a solution.
"""
function solve_problem(λ, μ, n, T, jset, vtj, jtv; δ=λ, ppn=30)
    p = (λ, μ, δ)
    u₀ = vcat(fill([ppn, 0], n)...)
    inf_v = rand(1:n)
    u₀[2*inf_v-1] -= 3
    u₀[2*inf_v] += 3
    dprob = DiscreteProblem(u₀, (0., T), p)
    rssa_jump_prob = JumpProblem(dprob, RSSA(), jset; vartojumps_map=vtj, jumptovars_map=jtv)
    return solve(rssa_jump_prob, SSAStepper())
end

function is_success(sol::ODESolution, allow_dieout::Bool)
    return allow_dieout || sum(sol[end][2:2:end]) > 0
end

"""
    to_tep(sol)

Convert the solution `sol` to an exact tep given by a two row matrix, where the first row
contains the time points and the second row contains the vertex indices.
"""
function to_tep(sol::ODESolution)
    @error "Not implemented"
end

"""
    to_tep(sol, dt)

Convert the solution `sol` to a sampled tep in the form of a matrix.
Each column corresponds to a time point and each row corresponds to a vertex.
"""
function to_tep(sol::ODESolution, dt::Real)
    return hcat([sol(t)[2:2:end] for t in 0:dt:sol.t[end]]...)
end

function plot_density(sol::ODESolution, ts, graph, g_name, j)
    densities = hcat([sol(t)[2:2:end] for t in ts]...)'
    return Plots.plot(ts, densities, title="Graph $(g_name), TEP $j"; legend=false)
end

end
