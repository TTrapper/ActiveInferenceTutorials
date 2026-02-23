import { Agent, Environment, GridRenderable, Position } from '../core/types';

/**
 * A 2D grid world environment where agents can move in 8 directions
 */
export class GridWorld implements Environment {
  size: number;
  agents: Agent[] = [];
  private walls: Set<string> = new Set();

  constructor(size: number) {
    this.size = size;
  }

  /**
   * Add a wall at the given position
   */
  addWall(position: Position): void {
    this.walls.add(`${position[0]},${position[1]}`);
  }

  /**
   * Remove a wall at the given position
   */
  removeWall(position: Position): void {
    this.walls.delete(`${position[0]},${position[1]}`);
  }

  /**
   * Check if there is a wall at the given position
   */
  isWall(position: Position): boolean {
    return this.walls.has(`${position[0]},${position[1]}`);
  }

  /**
   * Get all wall positions
   */
  getWalls(): Position[] {
    return Array.from(this.walls).map(s => s.split(',').map(Number) as Position);
  }

  /**
   * Clear all walls
   */
  clearWalls(): void {
    this.walls.clear();
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
   * Check if a position is valid within the grid (in bounds and not a wall)
   */
  isValidPosition(position: Position): boolean {
    return position[0] >= 0 && position[0] < this.size &&
           position[1] >= 0 && position[1] < this.size &&
           !this.isWall(position);
  }

  /**
   * Constrain a position to ensure it's within grid boundaries (no wrapping).
   * If the new position is a wall, it returns the currentPosition (staying put).
   */
  normalizePosition(position: Position, currentPosition?: Position): Position {
    const normalized: Position = [
      Math.max(0, Math.min(this.size - 1, position[0])),
      Math.max(0, Math.min(this.size - 1, position[1]))
    ];

    if (this.isWall(normalized) && currentPosition) {
      return [...currentPosition];
    }

    return normalized;
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

  /**
   * Convert a direction→probability policy into a 2D position grid.
   * Each direction probability is placed at the cell the source would
   * reach by moving in that direction.
   */
  policyToPositionGrid(
    policy: Map<string, number>,
    sourcePosition: Position
  ): number[][] {
    const grid = Array(this.size).fill(0).map(() =>
      Array(this.size).fill(0)
    );
    for (const [dirKey, prob] of policy) {
      const parts = dirKey.split(',').map(Number);
      const target = this.normalizePosition([
        sourcePosition[0] + parts[0],
        sourcePosition[1] + parts[1]
      ]);
      grid[target[0]][target[1]] += prob;
    }
    return grid;
  }

  /**
   * Print current grid state out to a string.
   * Pass an array of renderables to control what is shown;
   * defaults to all agents.
   */
  gridToString(items?: GridRenderable[]): string {
    const renderables = items ?? this.agents;
    let result = '';
    for (let i = 0; i < this.size; i++) {
      for (let j = 0; j < this.size; j++) {
        let cell = '.';
        if (this.isWall([j, i])) {
          cell = '#';
        } else {
          for (const item of renderables) {
            if (item.position[0] === j && item.position[1] === i) {
              cell = item.asciiSymbol;
              break;
            }
          }
        }
        result += cell;
      }
      result += '\n';
    }
    return result;
  }
}
