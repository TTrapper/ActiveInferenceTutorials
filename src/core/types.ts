/**
 * Core types used throughout the simulation framework
 */

/**
 * Position type representing a 2D coordinate
 */
export type Position = [number, number];

/**
 * Anything that can be rendered on the grid
 */
export interface GridRenderable {
  position: Position;
  asciiSymbol: string;
}

/**
 * Direction vectors for agent movement
 */
export const DIRECTION_VECTORS = [
  [0, 1],   // N
  [1, 1],   // NE
  [1, 0],   // E
  [1, -1],  // SE
  [0, -1],  // S
  [-1, -1], // SW
  [-1, 0],  // W
  [-1, 1]   // NW
] as const;

/**
 * Agent interface - the base for all agents in simulations
 */
export interface Agent {
  position: Position;
  asciiSymbol: string;
  perceive(): void;
  act(): void;
}

/**
 * Environment interface - the base for all simulation environments
 */
export interface Environment {
  step(): void;
  addAgent(agent: Agent): void;
  removeAgent(agent: Agent): boolean;
  isValidPosition(position: Position): boolean;
}

/**
 * SimulationState interface - what's passed to renderers
 */
export interface SimulationState {
  agents: {
    id: string;
    type: string;
    position: Position;
    [key: string]: any; // Additional properties based on agent type
  }[];
  environment: {
    type: string;
    size: number;
    [key: string]: any; // Additional properties based on environment type
  };
  // For predator-prey simulation
  predatorBelief?: number[][];
  preyMovementPolicy?: number[][];
  policyEditorActive?: boolean;
  // For state machine simulation
  transitionMatrix?: number[][];
  // General properties
  lessonType?: string;
  [key: string]: any; // Any additional simulation state
}

/**
 * StateChangeListener type - for subscribing to simulation updates
 */
export type StateChangeListener = (state: SimulationState) => void;

/**
 * Simulation controller interface - manages the simulation lifecycle
 */
export interface SimulationController {
  start(intervalMs?: number): void;
  pause(): void;
  reset(): void;
  step(): void;
  getState(): SimulationState;
  addStateChangeListener(listener: StateChangeListener): void;
}