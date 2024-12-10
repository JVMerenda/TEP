Base.Vector{Float64}(s::String) = Float64.([parse(Float64, item) for item in split(strip(s, ['[',']']), ",") if item != ""])

function parse_command_line_args()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--input"
        help = "Adjacency matrix of a graph as .npz file"
        arg_type = String
        default = ""

        "--N_teps"
        help = "Number of teps per graph"
        arg_type = Int64
        default = 1

        "--lambda"
        help = "Infection rate"
        arg_type = Float64
        default = 0.03

        "--mu"
        help = "Healing rate"
        arg_type = Float64
        default = 0.09

        "--T"
        help = "Time period"
        arg_type = Float64
        default = 100.0

        "--output"
        help = "Output directory"
        arg_type = String
        default = "."

        "--dt"
        help = "Sampling step; if nothing is given, the exact tep is returned"
        arg_type = Vector{Float64}
        default = Vector{Float64}()

        "--plot"
        help = "Plot the evolution of infectious density"
        action = :store_true

        "--allow-dieout"
        help = "Also store the result if the infection has died out by time `T`"
        action = :store_true

        "--use-msis"
        help = "Use the Metapopulation model with mobility MSIS"
        action = :store_true

        "--delta"
        help = "Mobility rate in MSIS"
        arg_type = Float64
        default = 0.1

        "--ppn"
        help = "Initial number of people per vertex in MSIS"
        arg_type = Int64
        default = 30
    end

    return parse_args(s)
end

function read_graph(f::AbstractString)
    read_single_graph = f -> split(f, "/")[end][1:end-4] => Graph(npzread(f))
    if isdir(f)
        return [read_single_graph(joinpath(f, g)) for g in readdir(f) if endswith(g, ".npz") && !startswith(g, "tep")]
    elseif endswith(f, ".npz")
        return [read_single_graph(f), ]
    else
        @warn "No graph detected at input location"
        return []
    end
end

function copy_graph(g, path)
    if isfile(path)
        a2 = npzread(path)
        @assert a2 == adjacency_matrix(g) "Different graphs with same name are given in the experiment"
        return false
    end
    npzwrite(path, adjacency_matrix(g))
    return true
end

"""
    geometric_graph(N, D, p; d=2)

Generates a geometric graph with `N` nodes in `d`-dimensional space.
This is a deterministic euclidean graph with cutoff distance `D`, followed by a rewiring of
one end of every edge with probability `p`.

# Arguments
- `N::Int`: Number of nodes in the graph.
- `D::Float64`: Distance cutoff for connecting nodes with an edge.
- `p::Float64`: Probability of rewiring each edge.
- `d::Int`: Dimensionality of the space (default is 2).

# Returns
- `Graph`: A geometric graph where nodes are connected based on Euclidean distance and rewiring probability.

# Description
This function creates an initial Euclidean graph with `N` nodes placed in `d` dimensions. Edges are formed between nodes that are within a distance `D` of each other. Each existing edge is then considered for rewiring with probability `p`, where one end of the edge is randomly fixed, and the other end is connected to a new randomly chosen node not already connected to the fixed node.
"""
function geometric_graph(N, D, p; d=2)
    g = euclidean_graph(N, d; cutoff=D)[1]
    for e in edges(g)
        if rand() < p
            fixed = rand([e.src, e.dst])
            rem_edge!(g, e)
            new = rand(setdiff(1:N, [fixed]))
            add_edge!(g, fixed, new)
        end
    end
    return g
end

"""
    discretize_distribution(dist::Distribution, support)

Evaluate the pdf of `dist` at support and return a discrete
distribution based on those normalized valies.
"""
function discretize_distribution(dist::Distribution, support)
    ps = [pdf(dist, x) for x in support]
    sps = sum(ps)
    return DiscreteNonParametric(support, ps ./ sps)
end

function unpack(params)
    return [p isa Distribution ? rand(p) : p for p in params]
end

function build_graphs(g_model, N_graphs, parameter_combinations, general_dir, g_name="graph")
    isdir(general_dir) || mkdir(general_dir)
    n_digits = length(string(N_graphs))
    for params in parameter_combinations
        output_dir = joinpath(general_dir, "N$(params.N)")
        isdir(output_dir) || mkdir(output_dir)
        for i in 1:N_graphs
            g = Graph(params[1])
            while !is_connected(g)
                g = g_model(unpack(params)...)
            end
            filename = joinpath(output_dir, "$(g_name)-$(lpad(i, n_digits, '0')).npz")
            if isfile(filename)
                @info "Graph already exists: $filename"
            else
                npzwrite(joinpath(output_dir, "$(g_name)-$(lpad(i, n_digits, '0')).npz"), adjacency_matrix(g))
            end
        end
    end
end
