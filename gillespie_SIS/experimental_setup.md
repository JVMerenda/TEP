## General remark

For all networks, except the real ones, the parameters are chosen such that the mean degree is (close to) 10.
The exact values can be found in their respective `build_graphs.jl` scripts.

Everything below is collected in the `run_experiments.sh` script.
Inside the script there are easy to adjust parameters for the number of threads, graphs, and teps.
```bash
bash run_experiments.sh
```
Or as a detached process
```bash
nohup bash run_experiments.sh > teps.o.log 2> teps.e.log&
```

There are two type of experiments. In SIS the nodes are individual agents that can be in one of two states: susceptible or infected.
In MSIS the nodes are groups of agents that can be in one of two states: susceptible or infected.
There is movement between the groups.
Agents van only infect individuals in the same group.
To use MSIS use the flag `--use-msis` and it is best to increase the infection rate to e.g. 0.3.

## The Erdos Renyi Graphs

The data in this folder has been generated using the following invocations of `generate_tep.jl`:


First generate the networks (see the file for details)
```bash
julia --project results/graphs/euclidean/build_graphs.jl 50
```
```bash
julia --project -t 10 generate_tep.jl --N_vertices 100  --input results/graphs/erdos-renyi/N100  --output results/sis/erdos-renyi/N100/  --N_teps 100
julia --project -t 10 generate_tep.jl --N_vertices 250  --input results/graphs/erdos-renyi/N250  --output results/sis/erdos-renyi/N250/  --N_teps 100
julia --project -t 10 generate_tep.jl --N_vertices 500  --input results/graphs/erdos-renyi/N500  --output results/sis/erdos-renyi/N500/  --N_teps 100
julia --project -t 10 generate_tep.jl --N_vertices 1000 --input results/graphs/erdos-renyi/N1000 --output results/sis/erdos-renyi/N1000/ --N_teps 100
```

Hence in each of the folders `N100`, `N500`, and `N1000` there are 10 graphs with 50 teps each.
The teps are stored as `tep-$i-$j.npz` files, where `$i` is the graph index and `$j` is the tep index.
The adjacency matrices of the graphs are stored as `graph-$i.npz` files.
For each network size the value of `p` is scaled such that the expected degree is 10.
The default values of 0.03 for the infection rate (`lambda`) and 0.09 for the healing rate (`mu`) are used.
The files can be read using the class provided in `read_tep.py`.

## Geometric and euclidean graphs

Given the applicability of the research, it should be tested wether the method can be used on geometric graphs.
In the euclidean case, the deterministic rule is kept.
Since Julia only provides deterministic methods for geometric graphs, rewiring with probability .1 is used to introduce randomness (in geometric).
The cutoff distance depends on the number of vertices such that the expected degree is around 10.
(N=100, cutoff=0.198), (N=500, cutoff=0.83), (N=1000, cutoff=0.58)

First generate the networks (see the file for details)
```bash
julia --project results/graphs/euclidean/build_graphs.jl 50
julia --project results/graphs/geometric/build_graphs.jl 50
```
Then generate the TEPs
```bash
julia --project -t 10 generate_tep.jl --input results/graphs/euclidean/N100/  --N_teps 100 --output results/sis/euclidean/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/euclidean/N250/  --N_teps 100 --output results/sis/euclidean/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/euclidean/N500/  --N_teps 100 --output results/sis/euclidean/N500/
julia --project -t 10 generate_tep.jl --input results/graphs/euclidean/N1000/ --N_teps 100 --output results/sis/euclidean/N1000/
```
```bash
julia --project -t 10 generate_tep.jl --input results/graphs/geometric/N100/  --N_teps 100 --output results/sis/geometric/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/geometric/N250/  --N_teps 100 --output results/sis/geometric/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/geometric/N500/  --N_teps 100 --output results/sis/geometric/N500/
julia --project -t 10 generate_tep.jl --input results/graphs/geometric/N1000/ --N_teps 100 --output results/sis/geometric/N1000/
```
## Regular

The regular graphs are generated using the following command
```bash
julia --project results/graphs/regular/build_graphs.jl 50
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/graphs/regular/N100/  --N_teps 100 --output results/sis/regular/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/regular/N250/  --N_teps 100 --output results/sis/regular/N250/
julia --project -t 10 generate_tep.jl --input results/graphs/regular/N500/  --N_teps 100 --output results/sis/regular/N500/
julia --project -t 10 generate_tep.jl --input results/graphs/regular/N1000/ --N_teps 100 --output results/sis/regular/N1000/
```

## Barabasi Albert

The Barabasi Albert graphs are generated using the following command
```bash
julia --project results/graphs/barabasi-albert/build_graphs.jl 50
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/graphs/barabasi-albert/N100/  --N_teps 100 --output results/sis/barabasi-albert/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/barabasi-albert/N250/  --N_teps 100 --output results/sis/barabasi-albert/N250/
julia --project -t 10 generate_tep.jl --input results/graphs/barabasi-albert/N500/  --N_teps 100 --output results/sis/barabasi-albert/N500/
julia --project -t 10 generate_tep.jl --input results/graphs/barabasi-albert/N1000/ --N_teps 100 --output results/sis/barabasi-albert/N1000/
```

## Scale Free

Static scale free graphs introduce less non-trivial structure than barabasi albert graphs and might thus provide a
clearer picture of the effects of the scale-free degree distribution.
The exponent is set to -2.5.

The Scale Free graphs are generated using the following command
```bash
julia --project results/graphs/scale-free/build_graphs.jl 50
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/graphs/scale-free/N100/  --N_teps 100 --output results/sis/scale-free/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/scale-free/N250/  --N_teps 100 --output results/sis/scale-free/N250/
julia --project -t 10 generate_tep.jl --input results/graphs/scale-free/N500/  --N_teps 100 --output results/sis/scale-free/N500/
julia --project -t 10 generate_tep.jl --input results/graphs/scale-free/N1000/ --N_teps 100 --output results/sis/scale-free/N1000/
```

## Watts Strogatz and grid

The rewiring probability is set to .1 in watts-strogatz and 0. in grid as a benchmark.

The Watts Strogatz graphs are generated using the following command
```bash
julia --project results/graphs/grid/build_graphs.jl 50
julia --project results/graphs/watts-strogatz/build_graphs.jl 50
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/graphs/grid/N100/  --N_teps 100 --output results/sis/grid/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/grid/N250/  --N_teps 100 --output results/sis/grid/N250/
julia --project -t 10 generate_tep.jl --input results/graphs/grid/N500/  --N_teps 100 --output results/sis/grid/N500/
julia --project -t 10 generate_tep.jl --input results/graphs/grid/N1000/ --N_teps 100 --output results/sis/grid/N1000/
```
```bash
julia --project -t 10 generate_tep.jl --input results/graphs/watts-strogatz/N100/  --N_teps 100 --output results/sis/watts-strogatz/N100/
julia --project -t 10 generate_tep.jl --input results/graphs/watts-strogatz/N250/  --N_teps 100 --output results/sis/watts-strogatz/N250/
julia --project -t 10 generate_tep.jl --input results/graphs/watts-strogatz/N500/  --N_teps 100 --output results/sis/watts-strogatz/N500/
julia --project -t 10 generate_tep.jl --input results/graphs/watts-strogatz/N1000/ --N_teps 100 --output results/sis/watts-strogatz/N1000/
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
julia --project results/graphs/real/convert_to_npz.jl
```
Then the TEPs can be generated
```bash
julia --project -t 10 generate_tep.jl --input results/graphs/real/infect-hyper  --N_teps 100 --output results/sis/real/infect-hyper
julia --project -t 10 generate_tep.jl --input results/graphs/real/sociopatterns --N_teps 100 --output results/sis/real/sociopatterns
```

## Legacy results
The results in the `N100_exact`, `N500_exact`, and `N1000_exact` folders have been generated using a previous version of the code.
They should not differ qualitatively from the current results, but the exact values might differ due to stochastic nature.
For reasons of reproducibility they are kept.
