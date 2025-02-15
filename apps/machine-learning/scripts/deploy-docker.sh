#!/bin/bash

# 1. Login to Azure Container Registry (if not already logged in)
az acr login --name leaguedraftv2registry

# 2. Build the Docker image
# Note: We specifically set platform to linux/amd64 since the container app runs on Azure
docker build --platform linux/amd64 \
  -t leaguedraftv2registry.azurecr.io/serve-model:latest \
  -f ./Dockerfile \
  ./

# 3. Push the image to Azure Container Registry
docker push leaguedraftv2registry.azurecr.io/serve-model:latest

# 4. Update the container app (optional - if you want to deploy immediately)
az containerapp update \
  --name leaguedraftv2inference \
  --resource-group LeagueDraftv2 \
  --image leaguedraftv2registry.azurecr.io/serve-model:latest \
  --set-env-vars "API_KEY=secretref:api-key" \
  --revision-suffix $(date +%Y%m%d%H%M)