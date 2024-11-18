## For Erdos Renyi Graphs

The data in this folder has been generated using the following invocations of `generate_tep.jl`:

```bash
julia --project -t 10 generate_tep.jl --p 0.1  --N_vertices 100  --output N100_exact/  --N_graphs 10 --N_teps 50
julia --project -t 10 generate_tep.jl --p 0.02 --N_vertices 500  --output N500_exact/  --N_graphs 10 --N_teps 50
julia --project -t 10 generate_tep.jl --p 0.01 --N_vertices 1000 --output N1000_exact/ --N_graphs 10 --N_teps 50
```

Hence in each of the folders `N100`, `N500`, and `N1000` there are 10 graphs with 50 teps each.
The teps are stored as `tep-$i-$j.npz` files, where `$i` is the graph index and `$j` is the tep index.
The adjacency matrices of the graphs are stored as `graph-$i.npz` files.
For each network size the value of `p` is scaled such that the expected degree is 10.
The default values of 0.03 for the infection rate (`lambda`) and 0.09 for the healing rate (`mu`) are used.
The files can be read using the class provided in `read_tep.py`.

Previously sampled TEPs have been generated in `N100_sampled/`, `N500_sampled/`, and `N1000_sampled/`.
The networks or the TEPs do not coincide.

## For other graphs