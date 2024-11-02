#!/bin/bash

# Activate virtual environment with hardcoded path
source "/home/azureuser/draftking-monorepo/apps/data-collection/.venv/bin/activate"

# Run the TypeScript script
exec yarn tsx ./src/scripts/extractToAzure.ts
