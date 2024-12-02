using Graphs
using NPZ

function to_vertex_index(edge_list)
    vertex_dict = Dict(v => i for (i, v) in enumerate(unique(edge_list)))
    vertex_map = v -> vertex_dict[v]
    return vertex_map.(edge_list)
end

function process_edge_list(filename, rows)
    edge_list = zeros(Int, countlines(filename), 2)
    open(filename) do f
        for (i, line) in enumerate(eachline(f))
            edge_list[i,:] = parse.(Int, split(line)[rows])
        end
    end
    edge_list = to_vertex_index(edge_list)
    graph = Graph(maximum(edge_list))
    for edge in eachrow(edge_list)
        add_edge!(graph, edge[1], edge[2])
    end
    return graph
end

function process_hyper_infect(filename="infect-hyper.edges")
    g_ih =  process_edge_list(filename, 1:2)
    isdir("infect-hyper") || mkdir("infect-hyper")
    npzwrite("infect-hyper/ih.npz", adjacency_matrix(g_ih))
end

function process_sociopatterns(dirname="sociopatterns_data", outputdir="sociopatterns")
    isdir(outputdir) || mkdir(outputdir)
    for file in readdir(dirname)
        g = process_edge_list(joinpath(dirname, file), 2:3)
        if length(connected_components(g)) == 1
            outputfile = join(split(file[1:end-4], "_")[2:4], "_") * ".npz"
            npzwrite(joinpath(outputdir, outputfile), adjacency_matrix(g))
            println("* $(outputfile): $(nv(g)) nodes, $(ne(g)) edges")
        else
            rm(joinpath(dirname, "$(file)"), force=true)
        end
    end
end

cd(@__DIR__)
process_hyper_infect()
process_sociopatterns()
