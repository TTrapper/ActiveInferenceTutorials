import { BaseSimulationController } from '../core/simulation';
import { SimulationState } from '../core/types';
import { GridWorld } from '../environments/gridworld';
import { PolicyPreyAgent } from '../agents/prey';
import { ActiveInferencePredator } from '../agents/predator';

/**
 * Lesson types for the predator-prey simulation
 */
export enum LessonType {
  /** Predator learns prey's per-position transitions (state = prey only) */
  LESSON_2 = 'lesson2',
  /** Joint state: predator position added to state key (state explosion) */
  LESSON_3 = 'lesson3'
}

// Fixed grid size for the predator-prey simulation
const FIXED_GRID_SIZE = 32;

/**
 * Predator-Prey simulation controller (Lessons 2 & 3)
 */
export class PredatorPreySimulation extends BaseSimulationController {
  predator: ActiveInferencePredator;
  prey: PolicyPreyAgent;
  gridWorld: GridWorld;
  lessonType: LessonType;

  constructor(lessonType: LessonType = LessonType.LESSON_2) {
    const gridWorld = new GridWorld(FIXED_GRID_SIZE);
    super(gridWorld);
    this.gridWorld = gridWorld;
    this.lessonType = lessonType;

    // Create prey first
    this.prey = new PolicyPreyAgent('prey1', [5, 5], gridWorld);
    gridWorld.addAgent(this.prey);

    // Create predator with temporary empty stateItems
    this.predator = new ActiveInferencePredator(
      'predator1',
      [25, 25],
      gridWorld,
      [] // placeholder — configured below
    );
    this.predator.setTargetAgent(this.prey);
    gridWorld.addAgent(this.predator);

    // Now configure stateItems for both agents
    this.configureStateItems();
  }

  /**
   * Set stateItems on prey and predator model based on current lesson
   */
  private configureStateItems(): void {
    if (this.lessonType === LessonType.LESSON_2) {
      // Predator model: state key = prey position only
      this.predator.setModelStateItems([this.prey]);
      // Prey behaviour: state key = its own position only
      this.prey.setStateItems([this.prey]);
    } else {
      // Lesson 3: joint state — both positions in the key
      this.predator.setModelStateItems([this.prey, this.predator]);
      this.prey.setStateItems([this.prey, this.predator]);
    }
  }

  getState(): SimulationState {
    // Prey's true policy for its current state
    const preyPolicy = this.prey.getCurrentPolicy();
    const preyTrueProbs = this.gridWorld.policyToPositionGrid(
      preyPolicy, this.prey.position
    );

    // Predator's learned model for the current state
    const stateItems = this.lessonType === LessonType.LESSON_2
      ? [this.prey]
      : [this.prey, this.predator];
    const stateKey = this.gridWorld.gridToString(stateItems);
    const modelPolicy =
      this.predator.preyModel.getMovementProbabilitiesForState(
        stateKey
      );
    const predatorModelProbs = this.gridWorld.policyToPositionGrid(
      modelPolicy, this.prey.position
    );

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
      predatorBelief: this.predator.preyBelief.map(row => [...row]),
      preyTrueProbs,
      predatorModelProbs,
      predatorVision: this.predator.perceive(false),
      lessonType: this.lessonType
    };
  }

  /**
   * Update the predator's vision range
   */
  setVisionRange(range: number): void {
    this.predator.setVisionRange(range);
    this.notifyStateChange();
  }

  reset(): void {
    this.pause();

    this.prey = new PolicyPreyAgent('prey1', [5, 5], this.gridWorld);

    this.predator = new ActiveInferencePredator(
      'predator1',
      [25, 25],
      this.gridWorld,
      []
    );
    this.predator.setTargetAgent(this.prey);

    this.gridWorld.clearAgents();
    this.gridWorld.addAgent(this.prey);
    this.gridWorld.addAgent(this.predator);

    this.configureStateItems();

    this.notifyStateChange();
  }
}
