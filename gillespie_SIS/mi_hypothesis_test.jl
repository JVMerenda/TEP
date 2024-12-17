using Graphs
using HypothesisTests
using NPZ
using Statistics
using ProgressMeter

graph_size = 100
dt = "0.10"
N_samples = 1000
f_test = mean
N_graphs = 50
N_teps = 100
p_val = 0.05

result_dir = joinpath(@__DIR__, "results", "sis")
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
        return FULL_TO_ABRV[full[1:end-13]] * "md"
    end
    return FULL_TO_ABRV[full]
end

function load_graph(result_dir, graph_size, graph_model, i_graph, g_pad_length=2)
    graph_dir = joinpath(result_dir, graph_model, "N$(graph_size)")
    graph_string = lpad(i_graph, g_pad_length, '0')
    return Graph(npzread(joinpath(graph_dir, "$(full_to_abrv(graph_model))-$(graph_string).npz")))
end

function load_mutual_info(
        result_dir, graph_size, graph_model, i_graph,
        j_tep, dt; g_pad_length=2, mi_pad_length=3
    )
    graph_dir = joinpath(result_dir, graph_model, "N$(graph_size)")
    graph_string = lpad(i_graph, g_pad_length, '0')
    mim_string = lpad(j_tep, mi_pad_length, '0')
    full_path = joinpath(graph_dir, "mim-$(full_to_abrv(graph_model))-$(graph_string)-$(mim_string)-$(dt).npz")
    return reshape(npzread(full_path)["M"], graph_size, graph_size)
end

function permutation_test(all_indices, graph_indices, mim; f=mean, N_samples=2500, p_val=0.05)
    full_sample = view(mim, all_indices)
    edge_sample = view(mim, graph_indices)
    test = ApproximatePermutationTest(edge_sample, full_sample, f, N_samples)
    return pvalue(test) <= p_val
end

function get_upper_traingular_indices(graph_size)
    indices = LinearIndices((1:graph_size, 1:graph_size))
    return [indices[i, j] for i in 1:graph_size for j in (i+1):graph_size]
end

function get_edge_indices(graph)
    indices = LinearIndices((1:nv(graph), 1:nv(graph)))
    return [indices[src(e), dst(e)] for e in edges(graph)]
end

all_indices = get_upper_traingular_indices(graph_size)
for graph_model in graph_models
    N_graphs_w_one = 0
    N_total_true = 0
    for i_graph in 1:N_graphs
        any_true = false
        graph = load_graph(result_dir, graph_size, graph_model, i_graph)
        edge_indices = get_edge_indices(graph)
        for j_tep in 1:N_teps
            try
                mutual_info = load_mutual_info(result_dir, graph_size, graph_model, i_graph, j_tep, dt)
                if permutation_test(all_indices, edge_indices, mutual_info; f=f_test, N_samples, p_val)
                    any_true = true
                    N_total_true += 1
                end
            catch
                @warn "Could not handle mutual information matrix for $(graph_model) $(i_graph) $(j_tep)"
                continue
            end
        end
        N_graphs_w_one += any_true
    end
    @info "For $(graph_model) $(N_total_true) / $(N_graphs*N_teps) permutation tests could be rejected. For $(N_graphs_w_one) / $(N_graphs) at least for one mutual information matrix the test could be rejected."
end

