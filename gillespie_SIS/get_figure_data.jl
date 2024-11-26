cd(@__DIR__)
include("src/GenerateTep.jl")
using .GenerateTep
using Graphs
using Plots
using Statistics
using DifferentialEquations
using Random
using DataFrames
using CSV

Random.seed!(3)

function get_sol_for_p(p; N=50, λ=0.025, μ=0.09, T=100.0)
    p = (
        N = N,
        p = p,
        λ = λ,
        μ = μ,
        T = T,
    )

    g = erdos_renyi(p.N, p.p)
    @assert length(connected_components(g)) == 1
    jset, vtj, jtv = generate_jump_sets(g)
    sol = solve_problem(p.λ, p.μ, p.N, p.T, jset, vtj, jtv)
    return g, sol
end

function get_mean_for_p(p; N=50, λ=0.025, μ=0.09, T=100.0)
    p = (
        λ = p*(N-1)*λ,
        μ = μ,
        T = T,
    )

    SIS_f!(di, i, p, t) = begin
        di .= p.λ * i[1] * (1. - i[1]) - p.μ * i[1]
    end

    prob = ODEProblem(SIS_f!, [1. / N], (0.0, p.T), p)
    sol = solve(prob, Tsit5(), saveat=0.05)
    return sol
end

function export_tep_as_list(filename, ts, tep)
    df = DataFrame(
        node = [i for i in axes(tep, 1) for j in axes(tep, 2)],
        time = [ts[j] for i in axes(tep, 1) for j in axes(tep, 2)],
        state = [Int64.(tep[i, j]) for i in axes(tep, 1) for j in axes(tep, 2)],
    )
    CSV.write(filename, df)
    return df
end

T = 100
dt = 1.
output_path = joinpath("text", "figures", "data")
ts = 0:dt:T

g, g_sol15 = get_sol_for_p(0.15; T)
m_sol15 = get_mean_for_p(0.15; T)
g, g_sol30 = get_sol_for_p(0.30; T)
m_sol30 = get_mean_for_p(0.30; T)

tep15 = to_tep(g_sol15, dt)
tep30 = to_tep(g_sol30, dt)
get_density = x -> [mean(c) for c in eachcol(x)]

df = DataFrame(
    t = ts,
    g15 = get_density(tep15),
    m15 = vcat(m_sol15(ts)...),
    g30 = get_density(tep30),
    m30 = vcat(m_sol30(ts)...),
)

CSV.write(joinpath(output_path, "densities.csv"), df)

export_tep_as_list(joinpath(output_path, "tep15.csv"), ts, tep15)
export_tep_as_list(joinpath(output_path, "tep30.csv"), ts, tep30)

plot(df.t, df.g15, label="Graph 0.15", xlabel="Time", ylabel="Density")
plot!(df.t, df.m15, label="Mean 0.15")
plot!(df.t, df.g30, label="Graph 0.30")
plot!(df.t, df.m30, label="Mean 0.30")
