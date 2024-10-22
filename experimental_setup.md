The data in this folder has been generated using the following invocations of `generate_tep.jl`:

```bash
julia --project -t 10 generate_tep.jl --N_vertices 100 --output N100/ --N_graphs 10 --N_teps 50 --dt [.01,.1,1.,]
julia --project -t 10 generate_tep.jl --N_vertices 500 --output N500/ --N_graphs 10 --N_teps 50 --dt [.01,.1,1.,]
julia --project -t 10 generate_tep.jl --N_vertices 1000 --output N1000/ --N_graphs 10 --N_teps 50 --dt [.01,.1,1.,]
```

Hence in each of the folders `N100`, `N500`, and `N1000` there are 10 graphs with 50 teps each, sampled at time steps of 0.01, 0.1, 1.0.
The teps are stored as `tep-$i-$j-$dt.npz` files, where `$i` is the graph index and `$j` is the tep index.
The adjacency matrices of the graphs are stored as `graph-$i.npz` files.
