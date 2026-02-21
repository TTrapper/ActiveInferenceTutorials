import { Agent, Position } from '../core/types';
import { GridWorld } from '../environments/gridworld';

/**
 * A simple agent that transitions between states. We use the grid locations as
   possible states and the agent's position as the index of the current state.
 */
export class StateMachineAgent implements Agent {
  id: string;
  position: Position; // We use position as a state index
  asciiSymbol = 'S';
  environment: GridWorld; // Each location on the grid represents a possile state
  transitionMatrix: number[][]; // Probability matrix for state transitions

  constructor(id: string, position: Position, environment: GridWorld) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;

    // Create a random transition matrix
    // For an NxN grid, we have N^2 possible states
    const worldSize = environment.size;
    const numStates = worldSize * worldSize;
    this.transitionMatrix = this.createTransitionMatrix(numStates);
  }

  /**
   * Creates a random transition matrix for state transitions
   */
  private createTransitionMatrix(numStates: number): number[][] {
    const matrix: number[][] = [];
    
    // Create random probabilities for each state
    for (let i = 0; i < numStates; i++) {
      const row: number[] = [];
      
      // Generate random values for transitions
      for (let j = 0; j < numStates; j++) {
        row.push(Math.random());
      }
      
      // Normalize the row so probabilities sum to 1
      const rowSum = row.reduce((sum, val) => sum + val, 0);
      const normalizedRow = row.map(val => val / rowSum);
      
      matrix.push(normalizedRow);
    }
    
    return matrix;
  }

  /**
   * Convert 2D position to a state index
   */
  private positionToStateIndex(position: Position): number {
    const worldSize = this.environment.size;
    return position[0] * worldSize + position[1];
  }

  /**
   * Convert state index to 2D position
   */
  private stateIndexToPosition(stateIndex: number): Position {
    const worldSize = this.environment.size;
    const x = Math.floor(stateIndex / worldSize);
    const y = stateIndex % worldSize;
    return [x, y];
  }

  /**
   * Sample the next state based on transition probabilities
   */
  private sampleNextState(currentStateIndex: number): number {
    const probabilities = this.transitionMatrix[currentStateIndex];
    
    // Use weighted random selection
    const randomValue = Math.random();
    let cumulativeProbability = 0;
    
    for (let i = 0; i < probabilities.length; i++) {
      cumulativeProbability += probabilities[i];
      if (randomValue <= cumulativeProbability) {
        return i;
      }
    }
    
    // If we somehow get here (shouldn't happen if probabilities sum to 1)
    return 0;
  }

  /**
   * Perception is a no-op for the state machine
   */
  perceive(): void {
    // State machine doesn't need perception
  }

  /**
   * Move between states according to the transition matrix
   */
  act(): void {
    // Get current state index
    const currentStateIndex = this.positionToStateIndex(this.position);
    
    // Sample next state
    const nextStateIndex = this.sampleNextState(currentStateIndex);
    
    // Convert to position
    const newPosition = this.stateIndexToPosition(nextStateIndex);

    // Apply the move with normalization (wrapping around grid boundaries)
    this.position = this.environment.normalizePosition(newPosition);
  }
}
