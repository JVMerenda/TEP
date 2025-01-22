#!/bin/bash

# For msis
# DYN_DIR="msis"
# LAMBDA=0.3
# MSIS_FLAG="--use-msis"
# DT_FLAG="--dt 0.1"

# For sis
DYN_DIR="sis"
LAMBDA=0.03
MSIS_FLAG=""
DT_FLAG=""

N_THREADS=6
N_GRAPHS=50
N_TEPS=10

OUTPUT_DIR="results"

GRAPH_TYPES=("erdos-renyi" "regular" "barabasi-albert" "scale-free" "watts-strogatz" "euclidean" "geometric" "erdos-renyi-multi-degree" "regular-multi-degree" "barabasi-albert-multi-degree" "scale-free-multi-degree" "watts-strogatz-multi-degree" "euclidean-multi-degree" "grid-multi-degree" "geometric" "geometric-multi-degree") # real excluded, since they are handled differently

NODE_COUNTS=(100 200 250 300 400 500 600 700 750 800 900 1000)

echo "Generating graphs... ($N_GRAPHS per model)"

for graph in "${GRAPH_TYPES[@]}"; do
    julia --project "results/graphs/${graph}/build_graphs.jl" "$N_GRAPHS"
done
julia --project results/graphs/real/convert_to_npz.jl
wait

# Loop through graph types and node counts
for n in "${NODE_COUNTS[@]}"; do
    echo ""
    echo "Running experiments for size ${n}"

    for graph in "${GRAPH_TYPES[@]}"; do
        echo ""
        echo "Running experiments for ${graph} (size ${n})"
        julia --project -t "$N_THREADS" generate_tep.jl --lambda "$LAMBDA" --input "results/graphs/${graph}/N${n}/" --N_teps "$N_TEPS" --output "${OUTPUT_DIR}/${DYN_DIR}/${graph}/N${n}/" --store-mutual-info $(echo "${MSIS_FLAG}") $(echo "${DT_FLAG}")
    done
done

echo ""
echo "Running experiments for real networks"
julia --project -t "$N_THREADS" generate_tep.jl --lambda "$LAMBDA" --input results/graphs/real/infect-hyper --N_teps "$N_TEPS" --output "${OUTPUT_DIR}/${DYN_DIR}/real/infect-hyper" $(echo "${MSIS_FLAG}") $(echo "${DT_FLAG}")
julia --project -t "$N_THREADS" generate_tep.jl --lambda "$LAMBDA" --input results/graphs/real/sociopatterns --N_teps "$N_TEPS" --output "${OUTPUT_DIR}/${DYN_DIR}/real/sociopatterns" $(echo "${MSIS_FLAG}") $(echo "${DT_FLAG}")
