/**
 *
  LESSON TWO

  **Coding Challenge: Predator-Prey Simulation with Active Inference**

  **Scenario:**

  Imagine a 2D grid world where a predator is trying to catch prey. The prey moves randomly, and the predator uses active inference to navigate towards the prey.

  **Your Task:**

  1.  **Environment:** Create a 2D grid world. Represent the predator and prey as objects with positions on this grid.
  2.  **Prey Behavior:** Implement random movement for the prey. At each time step, the prey should move to a random adjacent cell (up, down, left, or right).
  3.  **Predator's Generative Model:** The predator has a generative model of the world. This model includes:
      *   Beliefs about the prey's movement patterns (e.g., random).
      *   Beliefs about its own movement capabilities (e.g., can move one cell in any direction).
  4.  **Predator's Free Energy:** The predator calculates its free energy based on the difference between its predictions (where it *thinks* the prey is) and its observations (where the prey actually *is*).
  5.  **Predator's Perception:** The predator updates its beliefs about the prey's location based on its observations.  This is a perceptual inference step.
  6.  **Predator's Action:** The predator chooses an action (moves to an adjacent cell) that it believes will minimize its free energy in the next time step.
  7.  **Simulation Loop:** Run the simulation for multiple time steps, updating the prey's position, the predator's beliefs, and the predator's position.
  8.  **Visualization:** (Optional, but highly recommended) Visualize the simulation. Show the grid world, the predator's position, the prey's position, and perhaps even the predator's beliefs about the prey's location (e.g., using a probability distribution heatmap).

  **Hints and Suggestions:**

  *   Start with a simple version of the generative model and gradually add complexity.
  *   Consider using libraries like NumPy for efficient array manipulation and Matplotlib or Pygame for visualization.
  *   Think about how the predator's beliefs about the prey's movement will influence its actions.
  *   Experiment with different ways of calculating free energy and updating beliefs.

  This coding challenge is more involved than the previous one, but it's a great opportunity to apply the concepts of active inference in a fun and interactive way.  Don't hesitate to ask if you have any questions or need further guidance. I'm here to support you throughout the process.  Let me know if you'd like to brainstorm specific implementation details or discuss any aspect of the challenge.
 *
 */

import { BaseSimulationController } from '../core/simulation';
import { SimulationState } from '../core/types';
import { GridWorld } from '../environments/gridworld';
import { RandomPreyAgent } from '../agents/prey';
import { ActiveInferencePredator } from '../agents/predator';

/**
 * Predator-Prey specific simulation controller
 */
export class PredatorPreySimulation extends BaseSimulationController {
  predator: ActiveInferencePredator;
  prey: RandomPreyAgent;
  gridWorld: GridWorld;

  constructor(gridSize: number = 8) {
    // Create the grid world environment
    const gridWorld = new GridWorld(gridSize);
    super(gridWorld);
    this.gridWorld = gridWorld;
    
    // Create and add prey
    this.prey = new RandomPreyAgent('prey1', [0, 0], gridWorld);
    gridWorld.addAgent(this.prey);
    
    // Create and add predator
    this.predator = new ActiveInferencePredator('predator1', [gridSize - 1, gridSize - 1], gridWorld);
    this.predator.setTargetAgent(this.prey);
    gridWorld.addAgent(this.predator);
  }

  /**
   * Get the current state of the simulation for visualization
   */
  getState(): SimulationState {
    return {
      agents: [
        {
          id: this.prey.id,
          type: 'prey',
          position: [...this.prey.position]
        },
        {
          id: this.predator.id,
          type: 'predator',
          position: [...this.predator.position]
        }
      ],
      environment: {
        type: 'gridworld',
        size: this.gridWorld.size
      },
      // Add predator's belief about prey location for visualization
      predatorBelief: this.predator.preyBelief.map(row => [...row])
    };
  }

  /**
   * Reset the simulation to initial state
   */
  reset(): void {
    this.pause();
    
    // Reset prey position
    this.prey.position = [0, 0];
    
    // Reset predator position and beliefs
    this.predator.position = [this.gridWorld.size - 1, this.gridWorld.size - 1];
    this.predator.preyBelief = Array(this.gridWorld.size).fill(0).map(() => 
      Array(this.gridWorld.size).fill(1 / (this.gridWorld.size * this.gridWorld.size))
    );

    this.notifyStateChange();
  }

  /**
   * Update simulation parameters
   */
  updateGridSize(size: number): void {
    if (size !== this.gridWorld.size) {
      this.gridWorld.size = size;
      this.reset();
    }
  }
}
