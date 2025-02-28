# Active Inference Tutorials

Interactive tutorials demonstrating active inference concepts through visualizations and coding challenges.

## Overview

This project contains a series of progressively complex simulations that demonstrate active inference principles. Each simulation is built using TypeScript and PixiJS for visualization.

### Current Simulations/Lessons

1. **State Transition Model (Lesson 1)**: A basic generative model that demonstrates state transitions based on probabilistic predictions.
2. **Predator-Prey Simulation (Lesson 2)**: A grid world where a predator agent uses active inference to hunt a prey agent that moves randomly.

## Getting Started

### Prerequisites

- Node.js (v16+)
- npm or yarn

### Installation

1. Clone the repository
2. Install dependencies:

```bash
npm install
```

### Running the Application

Start the development server:

```bash
npx parcel index.html
```

This will open the application in your browser at `http://localhost:1234`.

### Building for Production

```bash
npx parcel build index.html
```

This creates optimized files in the `dist` directory.

## Project Architecture

The framework is built with modular components that work together to create interactive simulations:

- **Agents**: Entities that perceive and act within an environment (StateMachineAgent, RandomPreyAgent, ActiveInferencePredator)
- **Environment**: The world in which agents operate (currently GridWorld)
- **Simulation Controllers**: Manage the lifecycle and interactions between agents and environment
- **Visualization**: Renders the simulation state using PixiJS

## Project Structure

- `/src/core`: Core interfaces and base classes
- `/src/agents`: Different agent implementations
- `/src/environments`: Simulation environments
- `/src/visualization`: Rendering and UI components
- `/src/simulations`: Specific simulation implementations

## Adding New Simulations

To add a new simulation:

1. Create relevant agents in the `/src/agents` directory
2. If needed, create a new environment in `/src/environments`
3. Create a simulation controller in `/src/simulations`
4. Add the new simulation to the UI by updating the `SimulationType` enum in `main.ts`

## License

MIT