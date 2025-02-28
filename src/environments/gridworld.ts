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
   * Normalize a position to ensure it's within grid boundaries (wrap around)
   */
  normalizePosition(position: Position): Position {
    return [
      (position[0] + this.size) % this.size,
      (position[1] + this.size) % this.size
    ];
  }

  /**
   * Execute one simulation step for all agents
   */
  step(): void {
    // First perception phase - all agents perceive
    for (const agent of this.agents) {
      agent.perceive();
    }
    
    // Then action phase - all agents act
    for (const agent of this.agents) {
      agent.act();
    }
  }
}