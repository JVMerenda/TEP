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
        (vs, vd, p, t) -> vs[1]*p[1], # rate
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
function solve_problem(λ, μ, n, T, jset, vtj, jtv)
    p = (λ, μ)
    u₀ = zeros(StateType, n)
    u₀[rand(eachindex(u₀))] = infectious
    dprob = DiscreteProblem(u₀, (0., T), p)
    rssa_jump_prob = JumpProblem(dprob, RSSA(), jset; vartojumps_map=vtj, jumptovars_map=jtv)
    return solve(rssa_jump_prob, SSAStepper())
end
