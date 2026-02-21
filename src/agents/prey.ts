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

  /** Items included when computing the state key (defaults to just this agent) */
  stateItems: GridRenderable[];

  /** Per-state movement policies, lazily generated on first visit */
  private statePolicies: Map<string, MovementPolicy> = new Map();

  constructor(id: string, position: Position, environment: GridWorld) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;
    this.stateItems = [this];
  }

  /**
   * Update the items used for state key computation.
   * Clears cached policies since the state space has changed.
   */
  setStateItems(items: GridRenderable[]): void {
    this.stateItems = items;
    this.statePolicies.clear();
  }

  /**
   * Get the current state key
   */
  private getStateKey(): string {
    return this.environment.gridToString(this.stateItems);
  }

  /**
   * Get the movement policy for a state, generating one if it doesn't exist yet
   */
  getPolicyForState(stateKey: string): MovementPolicy {
    if (!this.statePolicies.has(stateKey)) {
      this.statePolicies.set(stateKey, this.generateRandomPolicy());
    }
    return this.statePolicies.get(stateKey)!;
  }

  /**
   * Get the movement policy for the current state (for UI display)
   */
  getCurrentPolicy(): MovementPolicy {
    return this.getPolicyForState(this.getStateKey());
  }

  /**
   * Generate a random normalized movement policy over the 8 directions
   */
  private generateRandomPolicy(): MovementPolicy {
    const policy: MovementPolicy = new Map();
    let sum = 0;
    for (const dir of DIRECTION_VECTORS) {
      const weight = Math.random();
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
    const policy = this.getPolicyForState(this.getStateKey());

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
