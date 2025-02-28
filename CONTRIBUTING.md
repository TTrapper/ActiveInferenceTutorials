# Contributing to Active Inference Tutorials

Thank you for your interest in contributing to the Active Inference Tutorials project! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/ActiveInferenceTutorials.git`
3. Install dependencies: `npm install`
4. Start the development server: `npx parcel index.html`

## Project Organization

- All TypeScript code is in the `src/` directory
- The HTML interface is in `index.html`
- Use the existing module organization (agents, environments, etc.)

## Adding a New Lesson/Simulation

1. Create a new agent class if needed
2. Create a new simulation controller in `src/simulations/`
3. Update the `SimulationType` enum in `main.ts`
4. Add a description for your lesson

## Code Style

- Follow the TypeScript coding conventions in existing files
- Use 2 spaces for indentation
- Add JSDoc comments for all classes and methods
- Use interfaces for type definitions

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation to reflect any changes
3. Create a pull request with a clear description of the changes
4. Reference any related issues

## Contact

If you have questions or need help, please open an issue on the GitHub repository.