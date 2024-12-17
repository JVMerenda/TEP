using Graphs
using HypothesisTests
using NPZ
using Statistics

N_samples = 2500
f_test = mean
N_graphs = 50
N_teps = 100
p_val = 0.05

result_dir = pathjoin(@__DIR__, "results", "sis")
graph_models = readdir(result_dir)

const FULL_TO_ABRV = Dict(
    "erdos-renyi" => "er",
    "barabasi-albert" => "ba",
    "watts-strogatz" => "ws",
    "euclidean" => "euc",
    "geometric" => "geo",
    "grid" => "grid",
    "regular" => "reg",
    "scale-free" => "sf"
)

function full_to_abrv(full::AbstractString)
    if endswith(full, "-multi-degree")
        return FULL_TO_ABRV[abrv[1:end-13]] * "md"
    end
    return FULL_TO_ABRV[abrv]
end

function load_tep_graph(
        graph::AbstractString, N_vertices::Int, i_graph::Int, j_tep::Int, dt::Float64;
        graph_pad_length=1, tep_pad_length=2
    )
    padded_graph = lpad(i_graph, graph_pad_length, "0")
    padded_tep = lpad(j_tep, tep_pad_length , "0")
    base_dir = joinpath(RESULT_DIR, abrv_to_full(graph), "N$(N_vertices)")
    tep_location = joinpath(base_dir, "tep-$(graph)-$(padded_graph)-$(padded_tep)-$(dt).npz")
    graph_location = joinpath(base_dir, "$(graph)-$(padded_graph).npz")
    return npzread(tep_location), Graph(npzread(graph_location))
end

function load_graph(result_dir, graph_model, i_graph, g_pad_length=2)
    graph_dir = joinpath(result_dir, graph_model)
    graph_string = lpad(i_graph, g_pad_length)
    return Graph(npzread(joinpath(graph_dir, $(full_to_abrv(graph_model))-$(graph_string).npz)))
end

function load_mutual_info(
        result_dir, graph_model, i_graph, j_tep;
        g_pad_length=2, mi_pad_length=3
    )
    graph_dir = joinpath(result_dir, graph_model)
    graph_string = lpad(i_graph, g_pad_length)
    mim_string = lpad(j_tep, mi_pad_length)
    return npzread(graph_dir, "mim-$(full_to_abrv(graph_model))-$(graph_string)-$(mim_string)")
end

function permutation_test(graph, mim; f=mean, N_samples=2500, p_val=0.05)
    full_sample = mim[:]
    edge_sample = mim[adjacency_matrix[graph]][:]
    test = ApproximatePermutationTest(edge_sample, full_sample, f, N_samples)
    return pvalue(test) <= p_val
end

for graph_model in graph_models
    N_graphs_w_one = 0
    N_total_true = 0
    for i_graph in 1:N_graphs
        any_true = false
        graph = load_graph(result_dir, graph_model, i_graph)
        for j_mim in 1:N_teps
            mutual_info = load_mutual_info(result_dir, graph_model, i_graph, j_tep)
            if permutation_test(graph, mutual_info; f=f_test, N_samples, p_val)
                any_true = true
                N_total_true += 1
            end
        end
        N_graphs_w_one += any_true
    end
    @info "For $(graph_model) $(N_total_true) / $(N_graphs*N_teps) permutation tests could be rejected. For $(N_graphs_w_one) / $(N_graphs) at least for one mutual information matrix the test could be rejected."
end
