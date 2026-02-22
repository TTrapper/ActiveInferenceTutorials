import { Agent, GridRenderable, Position, DIRECTION_VECTORS } from '../core/types';
import { GridWorld } from '../environments/gridworld';

/**
 * Movement policy type for the prey agent.
 * Maps direction vector string (e.g. "1,0") to probability.
 */
export type MovementPolicy = Map<string, number>;

/**
 * A prey agent whose movement probabilities depend on the current grid state.
 * Each unique state (as returned by gridToString) has its own randomly generated
 * movement policy over the 8 direction vectors.
 *
 * The set of GridRenderables used to build the state key is configurable,
 * so later lessons can include predator position, walls, etc.
 */
export class PolicyPreyAgent implements Agent {
  id: string;
  position: Position;
  asciiSymbol = 'r';
  environment: GridWorld;

  /** Items included when computing the state key (for legacy/consistency) */
  stateItems: GridRenderable[];

  /** Agent to avoid (e.g. the predator) */
  private targetToAvoid: GridRenderable | null = null;

  constructor(id: string, position: Position, environment: GridWorld) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;
    this.stateItems = [this];
  }

  /**
   * Set the agent that this prey should try to avoid
   */
  setTargetToAvoid(agent: GridRenderable | null): void {
    this.targetToAvoid = agent;
  }

  /**
   * Update the items used for state key computation.
   */
  setStateItems(items: GridRenderable[]): void {
    this.stateItems = items;
  }

  /**
   * Get the movement policy for the current state.
   * Biases movement away from the targetToAvoid (or the center if none).
   */
  getCurrentPolicy(): MovementPolicy {
    const avoidPos = this.targetToAvoid 
      ? this.targetToAvoid.position 
      : [this.environment.size / 2 - 0.5, this.environment.size / 2 - 0.5] as Position;
    
    return this.generateAvoidancePolicy(avoidPos);
  }

  /**
   * Generate a policy that biases movement away from a given position
   */
  private generateAvoidancePolicy(avoidPos: Position): MovementPolicy {
    const policy: MovementPolicy = new Map();
    let sum = 0;

    for (const dir of DIRECTION_VECTORS) {
      const target = this.environment.normalizePosition([
        this.position[0] + dir[0],
        this.position[1] + dir[1]
      ]);

      // Calculate distance from target to the point being avoided
      const dist = Math.sqrt(
        Math.pow(target[0] - avoidPos[0], 2) + 
        Math.pow(target[1] - avoidPos[1], 2)
      );

      // Bias: higher weight if distance is larger
      // Using power of 4 for a strong "repulsion" effect
      const weight = Math.pow(dist, 4);
      policy.set(dir.toString(), weight);
      sum += weight;
    }

    // Normalize
    for (const [key, val] of policy) {
      policy.set(key, val / sum);
    }
    return policy;
  }

  perceive(): void {
    // Policy prey doesn't need perception
  }

  act(): void {
    const policy = this.getCurrentPolicy();

    // Sample a direction from the policy
    const random = Math.random();
    let cumulative = 0;
    let selectedMove: Position = [0, 0];

    for (const [dirKey, prob] of policy) {
      cumulative += prob;
      if (random <= cumulative) {
        const parts = dirKey.split(',').map(Number);
        selectedMove = [parts[0], parts[1]];
        break;
      }
    }

    const newPosition: Position = [
      this.position[0] + selectedMove[0],
      this.position[1] + selectedMove[1]
    ];

    this.position = this.environment.normalizePosition(newPosition);
  }
}
