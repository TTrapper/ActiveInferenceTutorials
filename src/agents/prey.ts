import { Agent, Position } from '../core/types';
import { GridWorld } from '../environments/gridworld';

/**
 * A simple prey agent that moves randomly
 */
export class RandomPreyAgent implements Agent {
  id: string;
  position: Position;
  environment: GridWorld;

  constructor(id: string, position: Position, environment: GridWorld) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;
  }

  /**
   * Perception is a no-op for random prey
   */
  perceive(): void {
    // Random prey doesn't need perception
  }

  /**
   * Move the prey randomly in one of the possible directions
   */
  act(): void {
    // Choose random x and y movement (-1, 0, or 1)
    const xMove = Math.floor(Math.random() * 3) - 1;
    const yMove = Math.floor(Math.random() * 3) - 1;

    // Calculate new position
    const newPosition: Position = [
      this.position[0] + xMove,
      this.position[1] + yMove
    ];

    // Apply the move with normalization (wrapping around grid boundaries)
    this.position = this.environment.normalizePosition(newPosition);
  }
}