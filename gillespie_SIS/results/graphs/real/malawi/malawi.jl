using CSV
using DataFrames
using DelimitedFiles
using Graphs

include(joinpath(@__DIR__, "..", "..", "..", "..", "src", "GenerateTep.jl"))
using .GenerateTep

function to_tep_and_graph(contacts; threshold=1)
    ts = 0:1:maximum(contacts.t)
    vs = 1:max(maximum(contacts.id1), maximum(contacts.id2))
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

malawi = CSV.read(joinpath(@__DIR__, "malawi.csv"), DataFrame)
@assert all(malawi.contact_time .% 20 .== 0)
malawi.t = malawi.contact_time .÷ 20
minimum(malawi.id1)

tep, g = to_tep_and_graph(malawi)


mutual_info = mutual_information_matrix(tep; word_length=5)
writedlm(joinpath(@__DIR__, "malawi_dataset.csv"), vectorize_upper_triangular(mutual_info; include_diagonal=true)', ',')
writedlm(joinpath(@__DIR__, "malawi_labels.csv"), vectorize_upper_triangular(adjacency_matrix(g); include_diagonal=false)', ',')
