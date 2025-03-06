import { Agent, Position } from '../core/types';
import { GridWorld } from '../environments/gridworld';

/**
 * Movement policy type for the prey agent
 * A 5x5 grid of probabilities centered on the prey
 */
export type MovementPolicy = number[][];

/**
 * A prey agent with configurable movement policy
 */
export class PolicyPreyAgent implements Agent {
  id: string;
  position: Position;
  environment: GridWorld;
  movementPolicy: MovementPolicy = [];

  /**
   * Create a new prey agent with a configurable movement policy
   */
  constructor(id: string, position: Position, environment: GridWorld) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;

    // Initialize with a uniform movement policy (5x5 grid)
    this.initializeUniformPolicy();
  }

  /**
   * Reset movement policy to uniform distribution
   */
  initializeUniformPolicy(): void {
    // Create a 5x5 grid with uniform probabilities
    this.movementPolicy = Array(5).fill(0).map(() =>
      Array(5).fill(1)
    );
    this.normalizePolicy();
  }

  /**
   * Increment a specific cell in the movement policy
   * @param x X coordinate in policy grid (0-4)
   * @param y Y coordinate in policy grid (0-4)
   */
  incrementPolicyCell(x: number, y: number): void {
    if (x >= 0 && x < 5 && y >= 0 && y < 5) {
      this.movementPolicy[y][x] += 0.1;
      this.normalizePolicy();
    }
  }

  /**
   * Ensure the movement policy is normalized (sums to 1)
   */
  normalizePolicy(): void {
    let sum = 0;

    // Calculate sum
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        sum += this.movementPolicy[y][x];
      }
    }

    // Normalize if sum is greater than 0
    if (sum > 0) {
      for (let y = 0; y < 5; y++) {
        for (let x = 0; x < 5; x++) {
          this.movementPolicy[y][x] /= sum;
        }
      }
    } else {
      // If all probabilities are 0, set to uniform
      this.initializeUniformPolicy();
    }
  }

  /**
   * Get the movement vector from policy coordinates
   * Converts from policy grid (0-4) to movement (-2 to +2)
   */
  private getMovementFromPolicyCoord(x: number, y: number): Position {
    return [x - 2, y - 2]; // Center is (2,2) which maps to (0,0) movement
  }

  /**
   * Perception is a no-op for policy prey
   */
  perceive(): void {
    // Policy prey doesn't need perception
  }

  /**
   * Move the prey according to the configured policy
   */
  act(): void {
    // Choose a move based on policy probabilities
    const flattenedPolicy: { prob: number, x: number, y: number }[] = [];

    // Flatten the 2D policy into a 1D array with coordinates
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        if (this.movementPolicy[y][x] > 0) {
          flattenedPolicy.push({
            prob: this.movementPolicy[y][x],
            x,
            y
          });
        }
      }
    }

    // Randomly select a move based on probabilities
    const random = Math.random();
    let cumulativeProbability = 0;
    let selectedMove: Position = [0, 0]; // Default no movement

    for (const entry of flattenedPolicy) {
      cumulativeProbability += entry.prob;
      if (random <= cumulativeProbability) {
        selectedMove = this.getMovementFromPolicyCoord(entry.x, entry.y);
        break;
      }
    }

    // Calculate new position
    const newPosition: Position = [
      this.position[0] + selectedMove[0],
      this.position[1] + selectedMove[1]
    ];

    // Apply the move - normalize will keep it within boundaries
    this.position = this.environment.normalizePosition(newPosition);
  }
}

/**
 * Legacy RandomPreyAgent for backward compatibility
 */
export class RandomPreyAgent extends PolicyPreyAgent {
  constructor(id: string, position: Position, environment: GridWorld) {
    super(id, position, environment);
  }
}
