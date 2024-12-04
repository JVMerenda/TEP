## General remark

For all networks, except the real ones, the parameters are chosen such that the mean degree is (close to) 10.
The exact values can be found in their respective `build_graphs.jl` scripts.

Everything below is collected in the `run_experiments.sh` script.
Inside the script there are easy to adjust parameters for the number of threads, graphs, and teps.
```bash
bash run_experiments.sh
```

## The Erdos Renyi Graphs

The data in this folder has been generated using the following invocations of `generate_tep.jl`:

```bash
julia --project -t 10 generate_tep.jl --p 0.1  --N_vertices 100  --output results/erdos_renyi/N100/  --N_graphs 10 --N_teps 100
julia --project -t 10 generate_tep.jl --p 0.04 --N_vertices 250  --output results/erdos_renyi/N250/  --N_graphs 10 --N_teps 100
julia --project -t 10 generate_tep.jl --p 0.02 --N_vertices 500  --output results/erdos_renyi/N500/  --N_graphs 10 --N_teps 100
julia --project -t 10 generate_tep.jl --p 0.01 --N_vertices 1000 --output results/erdos_renyi/N1000/ --N_graphs 10 --N_teps 100
```

Hence in each of the folders `N100`, `N500`, and `N1000` there are 10 graphs with 50 teps each.
The teps are stored as `tep-$i-$j.npz` files, where `$i` is the graph index and `$j` is the tep index.
The adjacency matrices of the graphs are stored as `graph-$i.npz` files.
For each network size the value of `p` is scaled such that the expected degree is 10.
The default values of 0.03 for the infection rate (`lambda`) and 0.09 for the healing rate (`mu`) are used.
The files can be read using the class provided in `read_tep.py`.

## Geometric graphs

Given the applicability of the research, it should be tested wether the method can be used on geometric graphs.
Since Julia only provides deterministic methods for geometric graphs, rewiring with probability .1 is used to introduce randomness.
The cutoff distance depends on the number of vertices such that the expected degree is around 10.
(N=100, cutoff=0.198), (N=500, cutoff=0.83), (N=1000, cutoff=0.58)

First generate the networks (see the file for details)
```bash
julia --project results/geometric/build_graphs.jl 10
```
Then generate the TEPs
```bash
julia --project -t 10 generate_tep.jl --input results/geometric/N100/ --N_teps 100 --output results/geometric/N100/
julia --project -t 10 generate_tep.jl --input results/geometric/N250/ --N_teps 100 --output results/geometric/N100/
julia --project -t 10 generate_tep.jl --input results/geometric/N500/ --N_teps 100 --output results/geometric/N500/
julia --project -t 10 generate_tep.jl --input results/geometric/N1000/ --N_teps 100 --output results/geometric/N1000/
```
## Regular

The regular graphs are generated using the following command
```bash
julia --project results/regular/build_graphs.jl 10
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/regular/N100/ --N_teps 100 --output results/regular/N100/
julia --project -t 10 generate_tep.jl --input results/regular/N250/ --N_teps 100 --output results/regular/N250/
julia --project -t 10 generate_tep.jl --input results/regular/N500/ --N_teps 100 --output results/regular/N500/
julia --project -t 10 generate_tep.jl --input results/regular/N1000/ --N_teps 100 --output results/regular/N1000/
```

## Barabasi Albert

The Barabasi Albert graphs are generated using the following command
```bash
julia --project results/barabasi-albert/build_graphs.jl 10
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/barabasi-albert/N100/ --N_teps 100 --output results/barabasi-albert/N100/
julia --project -t 10 generate_tep.jl --input results/barabasi-albert/N250/ --N_teps 100 --output results/barabasi-albert/N250/
julia --project -t 10 generate_tep.jl --input results/barabasi-albert/N500/ --N_teps 100 --output results/barabasi-albert/N500/
julia --project -t 10 generate_tep.jl --input results/barabasi-albert/N1000/ --N_teps 100 --output results/barabasi-albert/N1000/
```

## Scale Free

Static scale free graphs introduce less non-trivial structure than barabasi albert graphs and might thus provide a
clearer picture of the effects of the scale-free degree distribution.
The exponent is set to -2.5.

The Scale Free graphs are generated using the following command
```bash
julia --project results/scale-free/build_graphs.jl 10
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/scale-free/N100/ --N_teps 100 --output results/scale-free/N100/
julia --project -t 10 generate_tep.jl --input results/scale-free/N250/ --N_teps 100 --output results/scale-free/N250/
julia --project -t 10 generate_tep.jl --input results/scale-free/N500/ --N_teps 100 --output results/scale-free/N500/
julia --project -t 10 generate_tep.jl --input results/scale-free/N1000/ --N_teps 100 --output results/scale-free/N1000/
```

## Watts Strogatz

The rewiring probability is set to .1.

The Watts Strogatz graphs are generated using the following command
```bash
julia --project results/watts-strogatz/build_graphs.jl
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/watts-strogatz/N100/ --N_teps 100 --output results/watts-strogatz/N100/
julia --project -t 10 generate_tep.jl --input results/watts-strogatz/N250/ --N_teps 100 --output results/watts-strogatz/N250/
julia --project -t 10 generate_tep.jl --input results/watts-strogatz/N500/ --N_teps 100 --output results/watts-strogatz/N500/
julia --project -t 10 generate_tep.jl --input results/watts-strogatz/N1000/ --N_teps 100 --output results/watts-strogatz/N1000/
```

## Real-life graphs

The following real-life graphs are available as well.

1. [Infect-Hyper](https://networkrepository.com/infect-hyper.php) 113 nodes w 2196 edges of human close proximity network.
2. [Infectious SocioPatterns dynamic contact networks](http://www.sociopatterns.org/datasets/infectious-sociopatterns-dynamic-contact-networks/) Collection of daily interactions within a museum. The daily graphs that result in a fully connected network are included. This results in:
    * 2009_05_03.npz: 305 nodes, 1847 edges
    * 2009_05_06.npz: 176 nodes, 745 edges
    * 2009_05_07.npz: 194 nodes, 801 edges
    * 2009_05_15.npz: 241 nodes, 1301 edges
    * 2009_05_16.npz: 241 nodes, 1504 edges
    * 2009_05_23.npz: 238 nodes, 1075 edges
    * 2009_06_06.npz: 142 nodes, 696 edges
    * 2009_06_07.npz: 155 nodes, 563 edges
    * 2009_06_14.npz: 138 nodes, 433 edges
    * 2009_07_04.npz: 127 nodes, 526 edges
    * 2009_07_09.npz: 114 nodes, 373 edges
    * 2009_07_15.npz: 410 nodes, 2765 edges
    * 2009_07_16.npz: 318 nodes, 1441 edges

First they have to be generated to the correct format from the edge list
```bash
julia --project results/real/convert_to_npz.jl
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/real/infect-hyper --N_teps 100 --output results/real/infect-hyper
julia --project -t 10 generate_tep.jl --input results/real/sociopatterns --N_teps 100 --output results/real/sociopatterns
```

## Legacy results
The results in the `N100_exact`, `N500_exact`, and `N1000_exact` folders have been generated using a previous version of the code.
They should not differ qualitatively from the current results, but the exact values might differ due to stochastic nature.
For reasons of reproducibility they are kept.
