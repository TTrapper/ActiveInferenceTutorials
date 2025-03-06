import { Agent, Environment, Position } from '../core/types';

/**
 * A 2D grid world environment where agents can move in 8 directions
 */
export class GridWorld implements Environment {
  size: number;
  agents: Agent[] = [];

  constructor(size: number) {
    this.size = size;
  }

  /**
   * Add an agent to the environment
   */
  addAgent(agent: Agent): void {
    this.agents.push(agent);
  }

  /**
   * Remove an agent from the environment
   */
  removeAgent(agent: Agent): boolean {
    const index = this.agents.indexOf(agent);
    if (index !== -1) {
      this.agents.splice(index, 1);
      return true;
    }
    return false;
  }

  /**
   * Check if a position is valid within the grid
   */
  isValidPosition(position: Position): boolean {
    return position[0] >= 0 && position[0] < this.size &&
           position[1] >= 0 && position[1] < this.size;
  }

  /**
   * Constrain a position to ensure it's within grid boundaries (no wrapping)
   */
  normalizePosition(position: Position): Position {
    // If the position is out of bounds, return the closest valid position
    return [
      Math.max(0, Math.min(this.size - 1, position[0])),
      Math.max(0, Math.min(this.size - 1, position[1]))
    ];
  }

  /**
   * Execute one simulation step for all agents
   */
  step(): void {

    // Action phase - all agents act
    for (const agent of this.agents) {
      agent.act();
    }

    // Perception phase - all agents perceive
    for (const agent of this.agents) {
      agent.perceive();
    }
  }

  /**
   * Remove all agents from the environment
   */
  clearAgents(): void {
    this.agents = [];
  }
}
