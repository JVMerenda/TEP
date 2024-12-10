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

## Graphs

See below for more information on the used graph models.
In general, there are two variants of each graph.
There is the basic one without suffix.
In this case the parameters are chosen such that each graph has a mean degree 10.

The second is denoted by the `-multi-degree` suffix (or `md` abbreviated).
Here the parameters that genrate the graph are sampled from a distribution, such that the degree sequences and other structure of each graph deviates more from the other the other graphs in the same model.

## The Erdos Renyi Graphs

First generate the networks (see the file for details)
```bash
julia --project results/graphs/euclidean/build_graphs.jl 50
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

## Regular

The regular graphs are generated using the following command
```bash
julia --project results/graphs/regular/build_graphs.jl 50
```

## Barabasi Albert

The Barabasi Albert graphs are generated using the following command
```bash
julia --project results/graphs/barabasi-albert/build_graphs.jl 50
```

## Scale Free

Static scale free graphs introduce less non-trivial structure than barabasi albert graphs and might thus provide a
clearer picture of the effects of the scale-free degree distribution.
The exponent is set to -2.5.

The Scale Free graphs are generated using the following command
```bash
julia --project results/graphs/scale-free/build_graphs.jl 50
```

## Watts Strogatz and grid

The rewiring probability is set to .1 in watts-strogatz and 0. in grid as a benchmark.

The Watts Strogatz graphs are generated using the following command
```bash
julia --project results/graphs/grid/build_graphs.jl 50
julia --project results/graphs/watts-strogatz/build_graphs.jl 50
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
