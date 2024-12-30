# @draftking/ui

A shared UI component library for the LoLDraftAI applications, built with React, TypeScript, and Tailwind CSS.

## Overview

This package contains reusable UI components and utilities used across the LoLDraftAI web and desktop applications. It provides a consistent design system and shared functionality for champion drafting, analysis, and team composition.

## Key Components

### Draft Components

- `AnalysisParent` - Main container for draft analysis functionality
- `BestChampionSuggestion` - Suggests optimal champion picks based on team composition
- `ChampionGrid` - Grid display of available champions with search and filtering
- `DraftAnalysis` - Analyzes current draft state and provides win probability
- `TeamPanel` - Displays team composition with champion slots
- `StatusMessage` - Shows current draft state and next picking team

### UI Components

- Customized shadcn/ui components (Button, Dialog, Toast, etc.)
- `AnimatedBeam` - Visual effect component for connections
- `WinrateBar` - Visualization component for win probabilities

## Core Features

- Champion selection and team composition
- Draft analysis and win probability calculation
- Champion role suggestions
- Persistent state management
- Responsive design
- Dark mode support

## Usage in Applications

### Web App (@draftking/web)

```tsx
import { ChampionGrid, DraftAnalysis } from "@draftking/ui";
// Used for the main web interface with standard browser APIs
```

### Desktop App (@draftking/desktop)

```tsx
import { ChampionGrid, DraftAnalysis } from "@draftking/ui";
// Used with Electron-specific storage and image handling
```

## Technical Details

### Dependencies

- React 18+
- Tailwind CSS
- shadcn/ui components
- Radix UI primitives
- Framer Motion

### Directory Structure

⌘K to generate a command
@README.md
Please suggest an improved version of the README.md file, it will mostly be used as a context for LLMs and Chatbots to understand the project. this readme.md is only for the web-frontend of my project.

here are some file contexts:
@TeamPanel.tsx @page.tsx @HelpModal.tsx @DraftAnalysisShowcase.tsx @DraftAnalysis.tsx @ChampionGrid.tsx @BestChampionSuggestion.tsx @api.ts @AnalysisParent.tsx @page.tsx @draftStore.ts @champions.ts @globals.css @globals.css @layout.tsx @not-found.tsx @page.tsx @package.json @next.config.ts

Do as in depth as possible for the Readme.
BTW, here is the readme of the machine learning package that is used where the ml model used by the frontend is trained@README.md

Here's an improved README.md for the web frontend:
Setup
bash
Install dependencies
npm install
Run development server
npm run dev
Build for production
npm run build
Start production server
npm start

### Project Structure

src/
├── app/
│ ├── draft/ # Draft analysis components
│ ├── stores/ # Global state management
│ └── api/ # API route handlers
├── components/ # Shared components
└── styles/ # Global styles
This README provides a comprehensive overview of the web frontend, its architecture, and its integration with the ML backend. It's structured to give both high-level understanding and detailed technical information, making it useful for developers and LLMs alike.
ok i think this part is superflous, i would prefer it to be focused on technical aspects but on a high level, don't need to document the actual code the code is there for that, so this part is not needed:

Now let's do the same exercice but for the desktop frontend. the main difference for the frontend is that for now it only has 1 page, and it allows live game tracking.

here are some files for the context of the desktop app:
@README.md @package.json @index.html @App.tsx @App.tsx @draftStore.ts @main.ts @preload.ts @electron-env.d.ts @PlainImage.tsx @Layout.tsx @DraftAnalysisShowcase.tsx @DraftAnalysis.tsx @BestChampionSuggestion.tsx @AnalysisParent.tsx

Please suggest a readme for the desktop app

Here's a focused technical README for the desktop application:
bash
Install dependencies
npm install
Run in development
npm run dev
Build for production
npm run build
Build and package (directory only)
npm run build:dir
This README focuses on the technical aspects and architecture while keeping implementation details in the code where they belong. It provides enough context for developers to understand the system without duplicating what's already in the source files.
nice. now let's do something similar to create a readme for the ui package, also explain how it is used in the 2 apps.
@AnalysisParent.tsx @BestChampionSuggestion.tsx @ChampionGrid.tsx @DraftAnalysis.tsx @DraftAnalysisShowcase.tsx @HelpModal.tsx @TeamPanel.tsx @animated-beam.tsx @button.tsx @toast.tsx @toaster.tsx @use-toast.ts @usePersistedState.ts @champions.ts @draftLogic.ts @types.ts @globals.css @package.json @hooks @config @lib @ui @WinrateBar.tsx @utils.ts @components.json @postcss.config.mjs @README.md @.eslintrc.js @tsconfig.json @StatusMessage.tsx @context-menu.tsx @dialog.tsx @input.tsx @navigation-menu.tsx @select.tsx @tooltip.tsx @champion_play_rates.json @tailwind.config.ts

I'll help create a comprehensive README.md for the UI package. Here's a suggested structure:
tsx
import { ChampionGrid, DraftAnalysis } from "@draftking/ui";
// Used for the main web interface with standard browser APIs

### Desktop App (@draftking/desktop)

tsx
import { ChampionGrid, DraftAnalysis } from "@draftking/ui";
// Used with Electron-specific storage and image handling

## Technical Details

### Dependencies

- React 18+
- Tailwind CSS
- shadcn/ui components
- Radix UI primitives
- Framer Motion

### Directory Structure

```
src/
components/
draftking/ # Domain-specific components
ui/ # Base UI components
hooks/ # Custom React hooks
lib/ # Utilities and types
config/ # Configuration files
types.ts # TypeScript types
```

### Configuration

- Tailwind CSS configuration
- PostCSS setup
- TypeScript configuration
- ESLint rules

## License

Private - All rights reserved
