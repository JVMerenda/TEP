#!/bin/bash

N_THREADS=10
N_GRAPHS=50
N_TEPS=100

GRAPH_TYPES=("regular" "barabasi-albert" "scale-free" "watts-strogatz" "euclidean" "grid" "geometric") # excluding erdos-renyi and real, since they are handled differently, geometric acted up, and has to be handled later
NODE_COUNTS=(100 250 500 1000)

echo "Generating graphs... ($N_GRAPHS per model)"

for graph in "${GRAPH_TYPES[@]}"; do
    julia --project "results/${graph}/build_graphs.jl" "$N_GRAPHS"&
done
julia --project results/real/convert_to_npz.jl
wait

# Loop through graph types and node counts
for n in "${NODE_COUNTS[@]}"; do
    echo ""
    echo "Running experiments for size ${n}"

    P_VAL=$(echo "scale=3; 10/$n" | bc)
    julia --project -t "$N_THREADS" generate_tep.jl --p "$P_VAL"  --N_vertices "$n"  --output results/erdos-renyi/N${n}/  --N_graphs "$N_GRAPHS" --N_teps "$N_TEPS"

    for graph in "${GRAPH_TYPES[@]}"; do
        echo ""
        echo "Running experiments for ${graph} (size ${n})"
        julia --project -t "$N_THREADS" generate_tep.jl --input "results/${graph}/N${n}/" --N_teps "$N_TEPS" --output "results/${graph}/N${n}/"
    done
done

echo ""
echo "Running experiments for real networks"
julia --project -t "$N_THREADS" generate_tep.jl --input results/real/infect-hyper --N_teps "$N_TEPS" --output results/real/infect-hyper
julia --project -t "$N_THREADS" generate_tep.jl --input results/real/sociopatterns --N_teps "$N_TEPS" --output results/real/sociopatterns
