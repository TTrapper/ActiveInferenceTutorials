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


  *******************************************************************************************************************************
  *******************************************************************************************************************************


  LESSON 3

  **Coding Challenge: Predator-Prey with Advanced Belief Updating**

  **Scenario**

  We'll keep the 2D grid world from the previous challenge, but now the predator will have a more advanced belief update mechanism.  Instead of simply diffusing its belief uniformly, the predator will use a Bayesian approach to update its belief based on its observations and its model of the prey's movement.

  **Your Task**

  1.  **Bayesian Belief Update**
      * Implement a Bayesian belief update rule for the predator.  This means that the predator's belief about the prey's location at the next time step will be a combination of:
          * Its prior belief (based on its model of the prey's movement).
          * Its likelihood (based on its current observation).

  2.  **Prey's Movement Model**
      * Define a more sophisticated movement model for the prey.  Instead of moving completely randomly, the prey could have some tendency to move in a particular direction or avoid certain areas of the grid.

  3.  **Predator's Generative Model**
      * Update the predator's generative model to incorporate the new prey movement model.

  4.  **Simulation and Visualization**
      * Run the simulation and visualize the predator's belief state over time.  Observe how the predator's belief changes as it gathers more information about the prey's movement patterns.

  **Hints and Suggestions**

  * For the Bayesian belief update, you can use the following formula:
      ```
      posterior_belief = likelihood * prior_belief / normalization_factor
      ```
      where the normalization factor ensures that the posterior belief is a valid probability distribution.

  * You can experiment with different prey movement models, such as:
      * Biased random walk (the prey has a higher probability of moving in a certain direction).
      * Avoidance behavior (the prey avoids certain areas of the grid, such as the edges or obstacles).

  * To visualize the predator's belief state, you can use a heatmap where the color intensity represents the probability of the prey being at that location.

  This challenge will help you understand how active inference agents can update their beliefs in a more sophisticated way, taking into account both their prior knowledge and their observations.  It's a step towards building more intelligent and adaptive agents that can learn and interact with their environment effectively.  Feel free to ask if you have any questions or need further guidance.  I'm here to support you as you explore this exciting area of research.
 *
 */

import { BaseSimulationController } from '../core/simulation';
import { SimulationState } from '../core/types';
import { GridWorld } from '../environments/gridworld';
import { PolicyPreyAgent, MovementPolicy } from '../agents/prey';
import { ActiveInferencePredator } from '../agents/predator';

/**
 * Lesson types for the predator-prey simulation
 */
export enum LessonType {
  LESSON_2 = 'lesson2', // Uniform belief distribution (no learning)
  LESSON_3 = 'lesson3'  // Advanced Bayesian belief update (with learning)
}

// Fixed grid size for the predator-prey simulation
const FIXED_GRID_SIZE = 32;

/**
 * Predator-Prey specific simulation controller
 */
export class PredatorPreySimulation extends BaseSimulationController {
  predator: ActiveInferencePredator;
  prey: PolicyPreyAgent;
  gridWorld: GridWorld;
  lessonType: LessonType;
  policyEditorActive: boolean = false;

  constructor(lessonType: LessonType = LessonType.LESSON_2) {
    // Create the grid world environment with fixed size
    const gridWorld = new GridWorld(FIXED_GRID_SIZE);
    super(gridWorld);
    this.gridWorld = gridWorld;
    this.lessonType = lessonType;

    // Create and add prey
    this.prey = new PolicyPreyAgent('prey1', [5, 5], gridWorld);
    gridWorld.addAgent(this.prey);

    // Create and add predator with the appropriate model based on lesson
    const useAdvancedModel = lessonType === LessonType.LESSON_3;
    this.predator = new ActiveInferencePredator(
      'predator1',
      [25, 25], // Position predator away from prey
      gridWorld,
      useAdvancedModel
    );
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
      predatorBelief: this.predator.preyBelief.map(row => [...row]),
      // Add the positions that the predator can see
      predatorVision: this.predator.perceive(false),
      // Add prey movement policy for policy editor
      preyMovementPolicy: this.prey.movementPolicy.map(row => [...row]),
      // Add lesson type for UI
      lessonType: this.lessonType,
      // Add policy editor active state
      policyEditorActive: this.policyEditorActive
    };
  }

  /**
   * Update the prey's movement policy
   * @param x X coordinate in policy grid (0-4)
   * @param y Y coordinate in policy grid (0-4)
   */
  updatePreyPolicy(x: number, y: number): void {
    this.prey.incrementPolicyCell(x, y);
    this.notifyStateChange();
  }

  /**
   * Reset prey's movement policy to uniform
   */
  resetPreyPolicy(): void {
    this.prey.initializeUniformPolicy();
    this.notifyStateChange();
  }

  /**
   * Toggle the policy editor active state
   */
  togglePolicyEditor(active?: boolean): void {
    if (active !== undefined) {
      this.policyEditorActive = active;
    } else {
      this.policyEditorActive = !this.policyEditorActive;
    }
    this.notifyStateChange();
  }

  /**
   * Reset the simulation to initial state
   */
  reset(): void {
    this.pause();

    // Re-initialize agents to ensure they have correct grid size
    // Create and add prey
    this.prey = new PolicyPreyAgent('prey1', [5, 5], this.gridWorld);

    // Create and add predator with the appropriate model based on lesson
    const useAdvancedModel = this.lessonType === LessonType.LESSON_3;
    this.predator = new ActiveInferencePredator(
      'predator1',
      [25, 25], // Position predator away from prey
      this.gridWorld,
      useAdvancedModel
    );
    this.predator.setTargetAgent(this.prey);

    // Update the environment agents
    this.gridWorld.clearAgents();
    this.gridWorld.addAgent(this.prey);
    this.gridWorld.addAgent(this.predator);

    this.notifyStateChange();
  }

  /**
   * Change the lesson type
   */
  setLesson(lessonType: LessonType): void {
    if (this.lessonType !== lessonType) {
      this.lessonType = lessonType;
      // Update the predator's generative model based on the lesson
      const useAdvancedModel = lessonType === LessonType.LESSON_3;
      this.predator.setGenerativeModel(useAdvancedModel);
      this.reset();
    }
  }
}
