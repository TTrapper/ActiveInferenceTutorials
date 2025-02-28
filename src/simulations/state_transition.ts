/**
 *
  LESSON ONE

  Learning Coach
  Great! I'm glad to hear you're a hands-on learner. Coding solutions from scratch is a fantastic way to solidify your understanding of active inference. I'll make sure to include coding challenges and exercises throughout our learning journey.

  Let's begin with the first subtopic: Theoretical Foundations.

  Active inference is built upon the idea that the brain is constantly trying to minimize its "free energy."  Think of free energy as a measure of how surprised the brain is by its sensory experiences. The brain does this by maintaining a generative model of the world, which is essentially a set of beliefs about how the world works. It then tries to predict what it will sense next, and if its predictions don't match reality, it updates its generative model to better align with the world.

  A key aspect of active inference is that it doesn't just passively receive sensory information; it actively seeks out information that will minimize its free energy. This is where the "active" part comes in.  For example, when you're looking for your keys, you don't just stand there waiting for them to magically appear. You actively move your head and eyes, exploring different locations until you find them. Your brain is using its generative model to predict where the keys might be, and then it's guiding your actions to test those predictions.

  Here's a simple coding challenge to get you started:

  Problem: Implement a basic generative model.

  Details:

  1. Choose a simple environment, such as a 1D grid world.
  2. Define a set of states for your environment (e.g., locations on the grid).
  3. Create a probabilistic model that describes the transitions between these states. This model represents your generative model.
  4. Write a function that takes the current state as input and samples a next state based on your transition probabilities. This function simulates how your generative model predicts the next state of the world.
  This exercise will help you understand how generative models work and how they can be used to make predictions about the world.  Don't worry about making it perfect, the goal is to start coding and experimenting with these ideas.
 *
 */


import { BaseSimulationController } from '../core/simulation';
import { SimulationState } from '../core/types';
import { GridWorld } from '../environments/gridworld';
import { StateMachineAgent } from '../agents/statemachine';

/**
 * State Transition Simulation showing a generative model in action
 */
export class StateTransitionSimulation extends BaseSimulationController {
  gridWorld: GridWorld;
  agent: StateMachineAgent;

  constructor(gridSize: number = 8) {
    // Create the grid world environment
    const gridWorld = new GridWorld(gridSize);
    super(gridWorld);
    this.gridWorld = gridWorld;
    
    // Create and add the state machine agent
    this.agent = new StateMachineAgent('machine1', [0, 0], gridWorld);
    gridWorld.addAgent(this.agent);
  }

  /**
   * Get the current state of the simulation for visualization
   */
  getState(): SimulationState {
    return {
      agents: [
        {
          id: this.agent.id,
          type: 'state_machine',
          position: [...this.agent.position]
        }
      ],
      environment: {
        type: 'gridworld',
        size: this.gridWorld.size
      },
      // We can add additional visualization data here if needed
      // For example, we could visualize the transition matrix
      transitionMatrix: this.agent.transitionMatrix.map(row => [...row])
    };
  }

  /**
   * Reset the simulation to initial state
   */
  reset(): void {
    this.pause();
    
    // Reset agent position to top-left corner
    this.agent.position = [0, 0];
    
    // Regenerate the transition matrix with the current grid size
    const numStates = this.gridWorld.size * this.gridWorld.size;
    this.agent.transitionMatrix = this.agent['createTransitionMatrix'](numStates);

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
