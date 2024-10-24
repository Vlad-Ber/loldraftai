#!/bin/bash

# Array of regions
regions=("EUW1" "KR" "NA1" "OC1")

# Array of scripts
scripts=("collectMatchIds.ts" "fetchPuuids.ts" "processMatches.ts" "updateLadder.ts")

# Create logs directory if it doesn't exist
mkdir -p logs

# Launch extractToAzure
yarn tsx "./src/scripts/extractToAzure.ts" > "./logs/log_extractToAzure.txt" 2>&1 &
sleep 30 # to avoid spikes in db usage

# Launch all scripts for all regions
for script in "${scripts[@]}"; do
    for region in "${regions[@]}"; do
        yarn tsx "./src/scripts/$script" --region "$region" > "./logs/log_${script%.ts}_${region}.txt" 2>&1 &
        echo "Started $script for $region"
        sleep 30 # to avoid spikes in db usage
    done
done

echo "All jobs started."