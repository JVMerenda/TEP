struct TEPLocation
    root::String
    network_type::String
    graph_id::String
    tep_id::String
end

struct TEP
    location::TEPLocation
    time_points::Vector{Float64}
    vertex_indices::Vector{Int}
    N_vertices::Int
    max_T::Float64
end

const ABRV_TO_FULL = Dict(
    "er" => "erdos-renyi",
    "ba" => "barabasi-albert",
    "ws" => "watts-strogatz",
    "euc" => "euclidean",
    "geo" => "geometric",
    "grid" => "grid",
    "reg" => "regular",
    "sf"  => "scale-free"
)

function abrv_to_full(abrv::String)::String
    if endswith(abrv, "md")
        return ABRV_TO_FULL[substring(abrv, 1, end-2)] * "-multi-degree"
    else
        return ABRV_TO_FULL[abrv]
    end
end

function load_tep(loc::TEPLocation)
    filename = get_tep_location(loc)
    data = npzread(filename)
    time_points = Vector{Float64}(data[:, 1])
    vertex_indices = Vector{Int}(Int.(data[:, 2]))
    N_vertices = maximum(vertex_indices)
    max_T = time_points[end]
    return TEP(loc, time_points, vertex_indices, N_vertices, max_T)
end

"""
    TEP(abrv::String, nb_vertices::Int, i_graph::Int, j_tep; nb_digits_g=2, nb_digits_tep=3)

Load the TEP with the given parameters.

# Arguments
- `abrv::String`: The abbreviation of the network type.
- `nb_vertices::Int`: The number of vertices in the network.
- `i_graph::Int`: The index of the graph.
- `j_tep::Int`: The index of the TEP.
- `nb_digits_g::Int=2`: The number of digits in the graph index.
- `nb_digits_tep::Int=3`: The number of digits in the TEP index.
"""
function TEP(abrv::String, nb_vertices::Int, i_graph::Int, j_tep::Int;
               nb_digits_g::Int=2, nb_digits_tep::Int=3)::TEP
    root = @sprintf("%s/%s/N%d", "/home/tim/Documents/overleaf/TEP/gillespie_SIS/results/sis",
                   abrv_to_full(abrv), nb_vertices)
    network_type = abrv
    graph_id = @sprintf("%0*d", nb_digits_g, i_graph)
    tep_id = @sprintf("%0*d", nb_digits_tep, j_tep)
    location = TEPLocation(root, network_type, graph_id, tep_id)
    return load_tep(location)
end

function get_tep_location(obj::TEPLocation)::String
    return "$(obj.root)/tep-$(obj.network_type)-$(obj.graph_id)-$(obj.tep_id).npz"
end
get_tep_location(obj::TEP)::String = get_tep_location(obj.location)

function get_graph_location(obj::TEPLocation)::String
    return "$(obj.root)/$(obj.network_type)-$(obj.graph_id).npz"
end
get_graph_location(obj::TEP)::String = get_graph_location(obj.location)

function get_sample_location(obj::TEPLocation, dt::Float64)::String
    return "$(obj.root)/tep-$(obj.network_type)-$(obj.graph_id)-$(obj.tep_id)-$(@sprintf("%.2f", dt)).npy"
end
get_sample_location(obj::TEP, dt::Float64)::String = get_sample_location(obj.location, dt)

function get_sample_location(obj::TEPLocation, dt::Float64, p::Float64)::String
    return "$(obj.root)/tep-$(obj.network_type)-$(obj.graph_id)-$(obj.tep_id)-$(@sprintf("%.2f", dt))-p$(@sprintf("%.3f", p)).npy"
end
get_sample_location(obj::TEP, dt::Float64, p::Float64)::String = get_sample_location(obj.location, dt, p)

function (obj::TEP)(t::Float64, p::Float64=0.0)::Vector{Int}
    if t < 0
        error("t must be non-negative.")
    end
    idx = searchsortedlast(obj.time_points, t)
    x = [count(x -> x == v, obj.vertex_indices[1:idx]) % 2 for v in 1:obj.N_vertices]

    if p > 0.0
        for i in 1:obj.N_vertices
            if rand() < p
                x[i] = 1 - x[i]
            end
        end
    end
    return x
end

function sample_at_ts(obj::TEP, ts::AbstractArray, p::Float64=0.0)
    return hcat([obj(t, p) for t in ts]...)'
end

function sample_with_dt(obj::TEP, dt::Float64, p::Float64=0.0)::Array{Int,2}
    ts = 0:dt:obj.max_T
    return sample_at_ts(obj, ts, p)
end

function sample(obj::TEP, t::Union{Float64, Vector{Float64}}, p::Float64=0.0)
    if isa(t, Float64)
        return sample_with_dt(obj, t, p)
    elseif isa(t, Vector{Float64})
        return sample_at_ts(obj, t, p)
    else
        error("Invalid type for t. Must be Float64 or Vector{Float64}.")
    end
end

function store_sample(obj::TEP, dt::Float64)
    data = sample_with_dt(obj, dt)
    filename = get_sample_location(obj, dt)
    Numpy.save(filename, data)
end

function store_sample_p(obj::TEP, dt::Float64, p::Float64)
    data = sample_with_dt(obj, dt, p)
    filename = get_sample_location(obj, dt, p)
    Numpy.save(filename, data)
end

function load_graph(obj::TEP)::Array{Float64,2}
    return npzread(get_graph_location(obj))
end

