#! /bin/bash

# Start the FastAPI server
uvicorn serve-model:app --host 0.0.0.0 --port=8000
