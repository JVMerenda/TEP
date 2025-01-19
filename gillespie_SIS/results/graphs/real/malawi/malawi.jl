using CSV
using DataFrames
using DelimitedFiles
using Graphs

include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep

function to_tep_and_graph(contacts, individuals; threshold=1)
    ts = 0:1:maximum(contacts.t)
    vs = individuals
    tep = zeros(Bool, length(ts), length(vs))
    contact_counts = zeros(Int64, length(vs), length(vs))
    for t in ts
        for row in eachrow(contacts[contacts.t .== t, :])
            contact_counts[row.id1, row.id2] += 1
            contact_counts[row.id2, row.id1] += 1
            tep[t+1, row.id1] = true
            tep[t+1, row.id2] = true
        end
    end
    g = Graph(contact_counts .>= threshold)
    return tep, g
end

function per_period(contacts, individuals, period_time; threshold=1)
    contacts.period = contacts.contact_time .÷ period_time

    teps = []
    gs = []

    for period in unique(contacts.period)
        period_contacts = contacts[contacts.period .== period, :]
        tep, g = to_tep_and_graph(period_contacts, individuals; threshold=threshold)
        push!(teps, tep)
        push!(gs, g)
    end
    return teps, gs
end

store_mutual_info = (name, data) -> writedlm(joinpath(@__DIR__, "$(name)_dataset.csv"), vectorize_upper_triangular(data; include_diagonal=true)', ',')
store_graph = (name, data) -> writedlm(joinpath(@__DIR__, "$(name)_labels.csv"), vectorize_upper_triangular(adjacency_matrix(data); include_diagonal=false)', ',')

npad = x -> lpad(string(x), 2, "0")
word_length = 5 # for mutual information calculation

malawi = CSV.read(joinpath(@__DIR__, "malawi.csv"), DataFrame)
individuals = sort(unique(vcat(malawi.id1, malawi.id2)))
@assert all(individuals .== 1:length(individuals))
@assert all(malawi.contact_time .% 20 .== 0)
malawi.t = malawi.contact_time .÷ 20

tep, g = to_tep_and_graph(malawi, individuals)
mutual_info = mutual_information_matrix(tep; word_length=5)
store_mutual_info("full", mutual_info)
store_graph("full", g)

seconds_per_day = 60 * 60 * 24
teps_per_day, gs_per_day = per_period(malawi, individuals, seconds_per_day)
for day in eachindex(teps_per_day)
    day_tep = teps_per_day[day]
    day_g = gs_per_day[day]
    day_mutual_info = mutual_information_matrix(day_tep; word_length)
    store_mutual_info("day_$(npad(day))", day_mutual_info)
    store_graph("day_$(npad(day))", day_g)
end

seconds_er_week = seconds_per_day * 7
teps_per_week, gs_per_week = per_period(malawi, individuals, seconds_er_week)
for week in eachindex(teps_per_week)
    week_tep = teps_per_week[week]
    week_g = gs_per_week[week]
    week_mutual_info = mutual_information_matrix(week_tep; word_length)
    store_mutual_info("week_$(npad(week))", week_mutual_info)
    store_graph("week_$(npad(week))", week_g)
end
