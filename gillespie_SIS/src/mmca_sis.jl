module MMCA

const StateType = Float64
const susceptible = zero(StateType)
const infectious = one(StateType)

using Graphs
using DifferentialEquations
using NetworkDynamics
using Statistics: mean
using Plots

"""
    generate_jump_set(graph)

In this case not really jump sets, but rather the network (vertex and edge) dynamics with which to simulate the SIS model.
"""
function generate_jump_sets(graph)
    vertex_f!(vn::AbstractArray{StateType}, v::AbstractArray{StateType}, nghbs, p, t) = begin
        # Probability of not being infected by any neighbor
        q = 1.0
        for nghb in nghbs
            q *= 1. - p[1] * nghb[1]
        end

        vn[1] = (1. - q) * (1 - v[1]) + (1 - p[2]) * v[1]
    end

    edge_f!(e, v_s, v_d, p, t) = e .= v_s

    ode_vertex = ODEVertex(f=vertex_f!, dim=1)
    ode_edge = StaticEdge(f=vertex_f!, dim=1)

    return network_dynamics(ode_vertex, ode_edge, g), nothing, nothing
end

function rate_to_prob(x, dt)
    return 1. - exp(-x * dt)
end

function termination_cb(; tol=1e-4)
    condition(u, t, int) = begin
        if t % 2 == 0 # Try winning time by not checking every timestep
            return all(i -> abs(u[i] - int.uprev[i]) < tol, eachindex(u))
        end

        return false
    end

    return DiscreteCallback(condition, terminate!, save_positions=(true, false))
end

function solve_problem(λ, μ, n, T, dynamics, vtj::Nothing, jtv::Nothing; δ=nothing, ppn=nothing, dt=0.1)
    p = [rate_to_prob(λ, dt), rate_to_prob(μ, dt)]
    u₀ = zeros(StateType, n)
    u₀[rand(eachindex(u₀))] = infectious
    prob = DiscreteProblem(dynamics, u₀, ( 0, Int(floor(T / dt)) ), p)
    return solve(prob, FunctionMap(); callback=termination_cb())
end

function is_success(sol::ODESolution, allow_dieout::Bool; tol=1e-4)
    return allow_dieout || any(u[1] >= tol for u in sol[end])
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
    return hcat(sol.u...)'
end

function plot_density(sol::ODESolution, ts, graph, g_name, j)
    densities = [mean(u) for u in sol.u]
    return Plots.plot(sol.t, densities, title="Graph $(g_name), TEP $j"; legend=false)
end

end
