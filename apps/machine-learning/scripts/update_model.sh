#!/bin/bash

# Exit on any error
set -e

# Configuration
REPO_PATH="/datadrive/draftking-monorepo"
ML_PATH="$REPO_PATH/apps/machine-learning"
VENV_PATH="$ML_PATH/venv"
PLAYRATES_PATH="$REPO_PATH/packages/ui/src/lib/config/champion_play_rates.json"
ACR_REGISTRY="leaguedraftv2registry.azurecr.io"
IMAGE_NAME="serve-model"

# Log function
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Change to project directory
cd $ML_PATH

# Update repository
log "Pulling latest changes"
git pull

# Activate virtual environment
log "Activating virtual environment"
source $VENV_PATH/bin/activate


# Update dependencies
log "Installing/updating dependencies"
pip install -r requirements-cpu.txt

# Run data pipeline
log "Downloading new data"
python ./scripts/match-prediction/download_data.py

log "Preparing data"
python ./scripts/match-prediction/prepare_data.py

log "Generating playrates"
python ./scripts/match-prediction/generate_playrates.py

# Commit and push playrates
log "Committing and pushing playrates"
git pull
git add $PLAYRATES_PATH
git commit -m "chore: update champion play rates [skip ci]" || log "No changes to playrates"
git push origin main || log "Failed to push playrates, continuing anyway"

# Train model
log "Training model"
python ./scripts/match-prediction/train.py

# Build and push Docker image
log "Building Docker image"
docker build -t $ACR_REGISTRY/$IMAGE_NAME:latest -f ./Dockerfile ./

log "Pushing Docker image"
docker push $ACR_REGISTRY/$IMAGE_NAME:latest

log "Script completed successfully"
