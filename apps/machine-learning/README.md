# League of Legends Match Prediction Model

## Overview

This repository contains a machine learning system designed to predict various aspects of League of Legends matches, including win probability and in-game statistics, based on team compositions. The model uses a combination of champion embeddings and game state features to make these predictions.

## Key Features

### Multi-Task Learning

The model simultaneously predicts multiple objectives:

- Win probability for team 1
- Game duration
- Per-player statistics at different timestamps (900s, 1200s, 1500s, 1800s):
  - KDA (Kills/Deaths/Assists)
  - Gold
  - Creep Score
  - Champion Level
  - Damage Stats (Physical/Magical/True)
- Team-wide objectives at timestamps:
  - Tower kills
  - Inhibitor kills
  - Baron/Dragon/Herald kills

### Model Architecture

- **Base Model**: Neural network with champion embeddings and categorical/numerical feature processing
- **Architecture Details**:
  - Champion embeddings (learned representations for each champion)
  - Categorical feature embeddings
  - Numerical feature projection
  - Multi-layer perceptron (MLP) for feature combination
  - Task-specific output heads

### Training Techniques

#### Strategic Masking

The model employs a sophisticated masking strategy during training to handle partial drafts:

- 20% chance: No masking (full draft visibility)
- 10% chance: Mask one full team (5 champions)
- 0.5% chance: Mask all champions (baseline predictions)
- 69.5% chance: Mask 1-9 champions with linear decay probability

#### Optimization

- AdamW optimizer with weight decay (0.01)
- OneCycleLR learning rate scheduler
- Gradient clipping
- Mixed precision training
- Gradient accumulation support
- Label smoothing for binary classification tasks

## Data Pipeline

### 1. Data Download (`download_data.py`)

- Downloads match data from Azure Blob Storage
- Supports incremental updates
- Parallel download implementation
- Configurable time window (default: last 3 months)

### 2. Data Preparation (`prepare_data.py`)

- Processes raw match data into training format
- Computes derived features
- Handles categorical encoding
- Performs train/test split
- Normalizes numerical features

### 3. Play Rate Generation (`generate_playrates.py`)

- Calculates champion play rates per role
- Tracks patch-specific statistics
- Outputs JSON for frontend consumption

## Validation System (`validation.py`)

Comprehensive validation system with subgroup analysis:

- ELO-based subgroups
- Patch-based subgroups
- Play rate-based subgroups (rare vs. common picks)
- Per-task performance metrics

## Configuration System

### Training Config (`config.py`)

Configurable parameters include:

- Model architecture (embedding dimensions, layers, etc.)
- Training parameters (learning rate, epochs, etc.)
- Masking strategy parameters
- Validation settings

### Task Definitions (`task_definitions.py`)

- Defines prediction tasks and their types
- Configures task weights for loss calculation
- Supports binary classification and regression tasks

## Technical Details

### Performance Optimizations

- CUDA support with automatic compilation
- Apple M1/M2 (MPS) support
- CPU fallback with optimized settings
- Automatic batch size and worker configuration
- Memory-efficient data loading

### Development Features

- Wandb integration for experiment tracking
- Profiling support
- Comprehensive logging
- Model checkpointing
- Graceful interruption handling

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Azure Storage Blob (for data download)
- Weights & Biases (optional, for experiment tracking)
