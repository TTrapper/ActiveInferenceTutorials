import { Agent, DIRECTION_VECTORS, GridRenderable, Position } from '../core/types';
import { GridWorld } from '../environments/gridworld';

/**
 * Interface for prey movement generative models
 */
export interface PreyGenerativeModel {
  /**
   * Update the model based on observed prey position
   */
  update(prey: Agent | null): void;

  /**
   * Get movement probability distribution
   * Returns a map of direction to probability
   */
  getMovementProbabilities(): Map<string, number>;

  /**
   * Reset the model to its initial state
   */
  reset(): void;
}

/**
 * Bayesian world model that learns per-state transition probabilities.
 *
 * The state key is built from a configurable set of GridRenderables
 * (e.g. just the prey, or [prey, predator]) so the same class serves
 * both Lesson 2 (prey position only) and Lesson 3 (joint state).
 *
 * Uses a Dirichlet-categorical model: each unique state key maps to a
 * set of direction counts initialised with a uniform prior (alpha=1).
 * After observing data the predictive distribution is:
 *
 *   P(m_new = i | D) = (alpha_i + C_i) / sum(alpha_j + C_j)
 */
export class BayesianWorldModel implements PreyGenerativeModel {
  private lastState: string | null = null;
  private lastPreyPosition: Position | null = null;
  private transitionCounts: Map<string, Map<string, number>> = new Map();

  constructor(
    private environment: GridWorld,
    private stateItems: GridRenderable[]
  ) {
    this.reset();
  }

  update(prey: Agent | null): void {
    if (prey === null) {
      this.lastPreyPosition = null;
      return;
    }
    const currentState = this.environment.gridToString(this.stateItems);
    if (this.lastState !== null && this.lastPreyPosition !== null) {
      const directionMoved = this.computeDirectionMoved(
        this.lastPreyPosition, prey.position
      );
      const dirKey = directionMoved.toString();
      if (!this.transitionCounts.has(this.lastState)) {
        this.transitionCounts.set(this.lastState, new Map());
      }
      const dirCounts = this.transitionCounts.get(this.lastState)!;
      dirCounts.set(dirKey, (dirCounts.get(dirKey) ?? 0) + 1);
    }
    this.lastState = currentState;
    this.lastPreyPosition = [...prey.position];
  }

  getMovementProbabilities(): Map<string, number> {
    const dirCounts = this.lastState !== null
      ? this.transitionCounts.get(this.lastState)
      : undefined;

    // No data for this state yet — return empty map
    if (!dirCounts || dirCounts.size === 0) {
      return new Map();
    }

    const total = Array.from(dirCounts.values())
      .reduce((sum, count) => sum + count, 0);

    const probs = new Map<string, number>();
    dirCounts.forEach((count, direction) => {
      probs.set(direction, count / total);
    });
    return probs;
  }

  reset(): void {
    this.transitionCounts.clear();
    this.lastState = null;
    this.lastPreyPosition = null;
  }

  private computeDirectionMoved(last: Position, current: Position): Position {
    const gridSize = this.environment.size;
    const dx = this.computeWrappedDifference(last[0], current[0], gridSize);
    const dy = this.computeWrappedDifference(last[1], current[1], gridSize);
    return [dx, dy];
  }

  private computeWrappedDifference(
    a: number, b: number, size: number
  ): number {
    let diff = b - a;
    if (diff > size / 2) {
      diff -= size;
    } else if (diff < -size / 2) {
      diff += size;
    }
    return diff;
  }
}

/**
 * A predator agent that uses active inference principles to hunt prey
 */
export class ActiveInferencePredator implements Agent {
  id: string;
  position: Position;
  asciiSymbol = 'P';
  environment: GridWorld;
  preyBelief: number[][];
  preyModel!: PreyGenerativeModel;
  targetAgent: Agent | null = null;
  visionRange: number;

  constructor(
    id: string,
    position: Position,
    environment: GridWorld,
    stateItems: GridRenderable[]
  ) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;
    this.visionRange = environment.size;

    // Initialize belief about prey's position to uniform distribution
    this.preyBelief = Array(environment.size).fill(0).map(() =>
      Array(environment.size).fill(
        1 / (environment.size * environment.size)
      )
    );

    this.preyModel = new BayesianWorldModel(environment, stateItems);
  }

  /**
   * Set the agent that this predator is hunting
   */
  setTargetAgent(agent: Agent): void {
    this.targetAgent = agent;
  }

  /**
   * Replace the predator's world model with one using new state items
   */
  setModelStateItems(stateItems: GridRenderable[]): void {
    this.preyModel = new BayesianWorldModel(this.environment, stateItems);
  }

  /**
   * Set the predator's vision range
   */
  setVisionRange(range: number): void {
    this.visionRange = Math.max(1, Math.min(range, this.environment.size));
  }

  /**
   * Perceive the environment and update beliefs about prey location
   */
  perceive(doUpdateBelief: boolean = true): Position[] {
    if (!this.targetAgent) return [];

    const perceivedPositions: Position[] = [];
    // Check all positions within vision range
    for (let i = -this.visionRange; i <= this.visionRange; i++) {
      for (let j = -this.visionRange; j <= this.visionRange; j++) {
        const newPos: Position = this.environment.normalizePosition([
          this.position[0] + i,
          this.position[1] + j
        ]);
        perceivedPositions.push(newPos);
      }
    }

    // Check if prey is within perceived positions
    let preyFound = false;
    for (const pos of perceivedPositions) {
      if (pos[0] === this.targetAgent.position[0] &&
          pos[1] === this.targetAgent.position[1]) {
        if (doUpdateBelief) {
          this.updateBelief(
            this.targetAgent.position, perceivedPositions
          );
        }
        preyFound = true;
        break;
      }
    }

    if (!preyFound) {
      this.updateBelief(null, perceivedPositions);
    }

    return perceivedPositions;
  }

  /**
   * Update belief about prey's NEXT location based on observation
   * and model for prey behavior
   */
  private updateBelief(
    preyPosition: Position | null,
    perceivedPositions: Array<Position>
  ): void {
    if (preyPosition) {
      // If prey is observed, update model and set probabilities
      for (let i = 0; i < this.environment.size; i++) {
        this.preyBelief[i].fill(0);
      }

      this.preyModel.update(this.targetAgent);

      const movementProbs = this.preyModel.getMovementProbabilities();

      movementProbs.forEach((probability, moveKey) => {
        const move: Position = this.arrayFromString(moveKey);
        const possiblePreyPos = this.environment.normalizePosition([
          preyPosition[0] + move[0],
          preyPosition[1] + move[1]
        ]);
        this.preyBelief[possiblePreyPos[0]][possiblePreyPos[1]] =
          probability;
      });
    } else {
      this.preyModel.update(null);
      for (const position of perceivedPositions) {
        this.preyBelief[position[0]][position[1]] = 0.0;
      }
      this.normalizePreyBelief();

      const newPreyBelief: number[][] = Array(this.environment.size)
        .fill(0.0)
        .map(() => Array(this.environment.size).fill(0));

      for (let i = 0; i < this.environment.size; i++) {
        for (let j = 0; j < this.environment.size; j++) {
          this.preyModel.getMovementProbabilities()
            .forEach((probability, moveKey) => {
              const move: Position = this.arrayFromString(moveKey);
              const possiblePreyPos: Position =
                this.environment.normalizePosition(
                  [i + move[0], j + move[1]]
                );
              newPreyBelief[possiblePreyPos[0]][possiblePreyPos[1]] +=
                probability * this.preyBelief[i][j];
            });
        }
      }
      this.preyBelief = newPreyBelief;
      this.normalizePreyBelief();
    }
  }

  /**
   * Samples one element from the options array based on the weights.
   */
  sampleFromWeights<T>(options: T[], weights: number[]): T | null {
    const totalWeight = weights.reduce(
      (sum, weight) => sum + weight, 0
    );
    if (totalWeight === 0) return null;
    let threshold = Math.random() * totalWeight;
    for (let i = 0; i < options.length; i++) {
      threshold -= weights[i];
      if (threshold <= 0) {
        return options[i];
      }
    }
    return options[options.length - 1] || null;
  }

  /**
   * Choose and execute action based on active inference principles
   */
  act(): void {
    const possibleMoves: Position[] = [];
    const moveWeights: number[] = [];

    for (let x = 0; x < this.preyBelief.length; x++) {
      for (let y = 0; y < this.preyBelief[x].length; y++) {
        possibleMoves.push([x, y]);
        moveWeights.push(this.preyBelief[x][y]);
      }
    }

    const selectedMove = this.sampleFromWeights(
      possibleMoves, moveWeights
    );
    if (selectedMove) {
      const targetPos: Position = [selectedMove[0], selectedMove[1]];
      const currentPos: Position = this.position;

      const dx = targetPos[0] - currentPos[0];
      const dy = targetPos[1] - currentPos[1];

      const stepX = Math.sign(dx);
      const stepY = Math.sign(dy);

      const newPos: Position = [
        currentPos[0] + stepX, currentPos[1] + stepY
      ];

      this.position = this.environment.normalizePosition(newPos);
    }
  }

  /**
   * Convert stringified array back to Position
   */
  arrayFromString(str: string): Position {
    return str.split(',').map(Number) as Position;
  }

  /**
   * Make preyBelief so probabilities sum to 1
   */
  normalizePreyBelief(): void {
    let sum = 0;
    for (let i = 0; i < this.environment.size; i++) {
      for (let j = 0; j < this.environment.size; j++) {
        sum += this.preyBelief[i][j];
      }
    }
    if (sum > 0) {
      for (let i = 0; i < this.environment.size; i++) {
        for (let j = 0; j < this.environment.size; j++) {
          this.preyBelief[i][j] /= sum;
        }
      }
    }
  }

  /**
   * Reset the predator's state
   */
  reset(): void {
    this.preyModel.reset();
    this.preyBelief = Array(this.environment.size).fill(0).map(() =>
      Array(this.environment.size).fill(
        1 / (this.environment.size * this.environment.size)
      )
    );
  }
}
