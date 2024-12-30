# LoLDraftAI Web Frontend

## Overview

LoLDraftAI's web frontend is a Next.js application that provides a sophisticated interface for League of Legends draft analysis. It leverages machine learning models to provide real-time draft predictions and champion recommendations.
it shares core components with the desktop application(@draftking/desktop) through a shared UI (@draftking/ui) package.

## Architecture

### Core Components

#### Draft Analysis System

- `TeamPanel`: Displays team compositions with interactive champion selection
- `ChampionGrid`: Provides filterable champion selection interface with favorites system
- `DraftAnalysis`: Renders win probability and detailed game predictions
- `BestChampionSuggestion`: Offers AI-powered champion recommendations based on current draft state

#### State Management

- Uses Zustand for global state management (`draftStore.ts`)
- Manages:
  - Current patch information
  - Draft state
  - Analysis results

#### API Integration

- RESTful communication with backend ML services
- Endpoints:
  - `/api/predict`: Basic win probability predictions
  - `/api/predict-in-depth`: Detailed game state predictions
  - `/api/metadata`: Model version and patch information

### Technical Stack

#### Core Technologies

- **Framework**: Next.js 15.0
- **Language**: TypeScript
- **Styling**: Tailwind CSS with custom theme system
- **UI Components**: Custom component library (@draftking/ui)

#### Key Dependencies

- `@radix-ui`: Accessible component primitives
- `zustand`: State management
- `next-themes`: Theme management
- `heroicons` & `react-icons`: Icon systems

## Features

### Draft Analysis

- Real-time win probability prediction
- Team composition analysis
- Champion synergy evaluation
- Role-specific recommendations

### User Interface

- Responsive design for all screen sizes
- Dark mode support
- Accessible component design
- Interactive champion selection
- Favorite champions system per role

### Data Visualization

- Win probability charts
- Champion impact analysis
- Gold difference predictions
- Team composition strength indicators

## Integration with ML Backend

### Model Communication

- Interfaces with a sophisticated ML model trained on millions of matches
- Supports multiple prediction types:
  - Win probability
  - Gold difference at 15 minutes
  - Champion impact scores
  - Team composition synergy

### Data Flow

1. User selects champions in draft
2. Frontend formats team composition data
3. API requests sent to ML backend
4. Results processed and displayed in real-time

## Development

### Setup

```bash
# Install dependencies
yarn install

# Run development server
yarn dev

# Build for production
yarn build

# Start production server
yarn start
```
