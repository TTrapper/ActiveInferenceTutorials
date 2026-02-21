import { BaseSimulationController } from '../core/simulation';
import { SimulationState } from '../core/types';
import { GridWorld } from '../environments/gridworld';
import { PolicyPreyAgent } from '../agents/prey';

// Fixed grid size
const FIXED_GRID_SIZE = 32;

/**
 * Lesson 1: Prey-only simulation
 *
 * A single prey agent moves on a 32x32 grid with state-dependent
 * (per-position) transition policies. No predator is present.
 * Demonstrates how a generative model defines state transitions.
 */
export class PreyOnlySimulation extends BaseSimulationController {
  prey: PolicyPreyAgent;
  gridWorld: GridWorld;

  constructor() {
    const gridWorld = new GridWorld(FIXED_GRID_SIZE);
    super(gridWorld);
    this.gridWorld = gridWorld;

    this.prey = new PolicyPreyAgent('prey1', [16, 16], gridWorld);
    gridWorld.addAgent(this.prey);
  }

  getState(): SimulationState {
    const preyPolicy = this.prey.getCurrentPolicy();
    const preyTrueProbs = this.gridWorld.policyToPositionGrid(
      preyPolicy, this.prey.position
    );

    return {
      agents: [
        {
          id: this.prey.id,
          type: 'prey',
          position: [...this.prey.position]
        }
      ],
      environment: {
        type: 'gridworld',
        size: this.gridWorld.size
      },
      preyTrueProbs,
      lessonType: 'lesson1'
    };
  }

  reset(): void {
    this.pause();

    this.prey = new PolicyPreyAgent('prey1', [16, 16], this.gridWorld);

    this.gridWorld.clearAgents();
    this.gridWorld.addAgent(this.prey);

    this.notifyStateChange();
  }
}
