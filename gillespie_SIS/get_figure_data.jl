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

get_density = x -> [mean(c) for c in eachcol(x)]

function get_sol_for_p(p, ts::AbstractRange; N=50, λ=0.025, μ=0.09, repeats=500, q=0.05)
    p = (
        N = N,
        p = p,
        λ = λ,
        μ = μ,
    )

    g = erdos_renyi(p.N, p.p)
    @assert length(connected_components(g)) == 1
    jset, vtj, jtv = generate_jump_sets(g)
    samples = Matrix{Float64}(undef, length(ts), repeats)
    row = 1
    ref_sol = solve_problem(p.λ, p.μ, p.N, ts[end], jset, vtj, jtv)
    while row <= repeats
        sol = solve_problem(p.λ, p.μ, p.N, ts[end], jset, vtj, jtv)
        if sum(u[1] for u in sol[end]) > 0.5
            samples[:, row] = [mean(u[1] for u in sol(t)) for t in ts]
            row += 1
        end
    end
    bounds = reduce(hcat, [quantile(samples[row, :], [q, 1 - q]) for row in axes(samples, 1)])'
    return g, ref_sol, bounds
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

g, g_sol15, bounds15 = get_sol_for_p(0.15, ts)
m_sol15 = get_mean_for_p(0.15; T)
g, g_sol30, bounds30 = get_sol_for_p(0.30, ts)
m_sol30 = get_mean_for_p(0.30; T)

tep15 = to_tep(g_sol15, dt)
tep30 = to_tep(g_sol30, dt)

df = DataFrame(
    t = ts,
    g15 = get_density(tep15),
    m15 = vcat(m_sol15(ts)...),
    lb15 = bounds15[:, 1],
    ub15 = bounds15[:, 2],
    g30 = get_density(tep30),
    m30 = vcat(m_sol30(ts)...),
    lb30 = bounds30[:, 1],
    ub30 = bounds30[:, 2],
)

CSV.write(joinpath(output_path, "densities.csv"), df)

export_tep_as_list(joinpath(output_path, "tep15.csv"), ts, tep15)
export_tep_as_list(joinpath(output_path, "tep30.csv"), ts, tep30)

plot(df.t, df.g15, label="Graph 0.15", xlabel="Time", ylabel="Density")
plot!(df.t, df.m15, label="Mean 0.15")
plot!(df.t, df.g30, label="Graph 0.30")
plot!(df.t, df.m30, label="Mean 0.30")
plot!(df.t, df.lb15, label="0.15 LB", linestyle=:dash)
plot!(df.t, df.ub15, label="0.15 UB", linestyle=:dash)
plot!(df.t, df.lb30, label="0.30 LB", linestyle=:dash)
plot!(df.t, df.ub30, label="0.30 UB", linestyle=:dash)
