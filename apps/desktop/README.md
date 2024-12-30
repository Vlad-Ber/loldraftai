# LoLDraftAI Desktop Application

## Overview

LoLDraftAI's desktop application is an Electron-based client that provides League of Legends draft analysis with live game integration. Built with Vite, React, and TypeScript, it shares core components with the web frontend(@draftking/web-frontend) through a shared UI (@draftking/ui) package.

## Technical Stack

### Core Technologies

- **Framework**: Electron + Vite + React
- **Language**: TypeScript
- **UI Components**: Shared @draftking/ui package
- **State Management**: Zustand
- **Storage**: electron-store
- **Auto Updates**: electron-updater

### Key Features

- Live game integration via League Client API
- Automatic champion draft tracking
- Offline-capable analysis
- Auto-updates system
- Cross-platform support (Windows/macOS)

## Architecture

### Electron Process Structure

- **Main Process**: Handles League client communication and system integration
- **Renderer Process**: React application for UI
- **Preload Scripts**: Secure bridge between processes

### League Client Integration

- Automatic client detection
- Real-time draft state monitoring
- Secure API communication via local endpoints
- Support for both Windows and macOS paths

### Data Persistence

- Local storage for user preferences
- Favorite champions per role
- ELO settings
- Current patch information

## Development

### Setup

```bash
# Install dependencies
yarn install

# Run in development
yarn dev

# Build for production
yarn build

# Build and package (directory only)
yarn build:dir
```

### Build Configuration

- Windows NSIS installer
- Auto-update support
- Icon assets included
- Production optimizations

### Distribution

- Automated releases via electron-builder
- Generic update server support
- Version management through package.json

## License

Proprietary - All Rights Reserved
