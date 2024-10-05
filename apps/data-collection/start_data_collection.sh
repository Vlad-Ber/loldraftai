#!/bin/bash

# Array of regions
regions=("EUW1" "KR" "NA1" "OC1")

# Array of scripts
scripts=("collectMatchIds.ts" "fetchPuuids.ts" "processMatches.ts" "updateLadder.ts")

# Launch all scripts for all regions
for script in "${scripts[@]}"; do
    for region in "${regions[@]}"; do
        yarn tsx "./src/scripts/$script" --region "$region" > "log_${script%.ts}_${region}.txt" 2>&1 &
        echo "Started $script for $region"
    done
done

echo "All jobs started. Use 'jobs' to list them."