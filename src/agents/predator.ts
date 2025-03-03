import { Agent, DIRECTION_VECTORS, Position } from '../core/types';
import { GridWorld } from '../environments/gridworld';

/**
 * Interface for prey movement generative models
 */
export interface PreyGenerativeModel {
  /**
   * Update the model based on observed prey position
   */
  update(preyPosition: Position): void;

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
 * Basic prey model with uniform movement probabilities (Lesson 2)
 */
export class UniformPreyModel implements PreyGenerativeModel {
  constructor(private environment: GridWorld) {}

  update(preyPosition: Position): void {
    // No learning in the uniform model
  }

  getMovementProbabilities(): Map<string, number> {
    const probs = new Map<string, number>();
    // Assign equal probability to all directions
    for (const dir of DIRECTION_VECTORS) {
      probs.set(dir.toString(), 1 / DIRECTION_VECTORS.length);
    }
    return probs;
  }

  reset(): void {
    // No state to reset
  }
}

/**
 * Bayesian prey model that learns movement patterns (Lesson 3)
 * For our discrete gridworld scenario where the prey can make one of several possible movements, we're dealing with a categorical distribution.

    Let's define:
    - $m$ as a possible movement from the set of all possible movements $M$
    - $\theta$ as the parameters of our distribution (probabilities of each movement)
    - $D$ as our observed data (sequence of movements)

    The Bayesian update follows:

    $$P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$$

    For a categorical distribution, the conjugate prior is a Dirichlet distribution:

    $$P(\theta) = \text{Dirichlet}(\theta|\alpha_1, \alpha_2, ..., \alpha_n)$$

    where $\alpha_i$ are the concentration parameters (or pseudocounts) for each movement.

    After observing data $D$ with counts $C_1, C_2, ..., C_n$ for each movement, the posterior is:

    $$P(\theta|D) = \text{Dirichlet}(\theta|\alpha_1 + C_1, \alpha_2 + C_2, ..., \alpha_n + C_n)$$

    The predictive distribution for a new movement $m_{new}$ given previous observations is:

    $$P(m_{new}=i|D) = \frac{\alpha_i + C_i}{\sum_{j=1}^{n}(\alpha_j + C_j)}$$

    If you start with a uniform prior where all $\alpha_i = 1$ (which is often called a "flat" prior), then the predictive probability simplifies to:

    $$P(m_{new}=i|D) = \frac{1 + C_i}{n + \sum_{j=1}^{n}C_j}$$

    As the predator collects more observations, this converges to the maximum likelihood estimate:

    $$P(m_{new}=i|D) \approx \frac{C_i}{\sum_{j=1}^{n}C_j}$$

    This is - counting occurrences and normalizing to get probabilities.
 */
export class BayesianPreyModel implements PreyGenerativeModel {
  private lastPreyPosition: Position | null = null;
  private movementCounts: Map<string, number> = new Map();

  constructor(private environment: GridWorld) {
    this.reset();
  }

  update(preyPosition: Position): void {
    if (this.lastPreyPosition !== null) {
      const directionMoved = this.computeDirectionMoved(this.lastPreyPosition, preyPosition);
      const currentCount = this.movementCounts.get(directionMoved.toString()) ?? 0;
      this.movementCounts.set(directionMoved.toString(), currentCount + 1);
    }
    this.lastPreyPosition = [...preyPosition];
  }

  getMovementProbabilities(): Map<string, number> {
    const totalCounts = Array.from(this.movementCounts.values()).reduce((sum, count) => sum + count, 0);
    const probs = new Map<string, number>();

    this.movementCounts.forEach((count, direction) => {
      probs.set(direction, count / totalCounts);
    });

    return probs;
  }

  reset(): void {
    this.lastPreyPosition = null;
    this.movementCounts.clear();

    // Initialize with uniform prior
    for (const dir of DIRECTION_VECTORS) {
      this.movementCounts.set(dir.toString(), 1);
    }
  }

  private computeDirectionMoved(last: Position, current: Position): Position {
    const gridSize = this.environment.size;
    const dx = this.computeWrappedDifference(last[0], current[0], gridSize);
    const dy = this.computeWrappedDifference(last[1], current[1], gridSize);
    return [dx, dy];
  }

  private computeWrappedDifference(a: number, b: number, size: number): number {
    let diff = b - a;
    // Adjust if the difference is larger than half the grid size (wrap-around)
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
  environment: GridWorld;
  preyBelief: number[][];
  preyModel: PreyGenerativeModel;
  targetAgent: Agent | null = null;

  constructor(id: string, position: Position, environment: GridWorld, useAdvancedModel: boolean = false) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;

    // Initialize belief about prey's position to uniform distribution
    this.preyBelief = Array(environment.size).fill(0).map(() =>
      Array(environment.size).fill(1 / (environment.size * environment.size))
    );

    // Create the appropriate generative model based on the lesson
    this.preyModel = useAdvancedModel
      ? new BayesianPreyModel(environment)
      : new UniformPreyModel(environment);
  }

  /**
   * Set the agent that this predator is hunting
   */
  setTargetAgent(agent: Agent): void {
    this.targetAgent = agent;
  }

  /**
   * Change the predator's generative model
   */
  setGenerativeModel(useAdvancedModel: boolean): void {
    this.preyModel = useAdvancedModel
      ? new BayesianPreyModel(this.environment)
      : new UniformPreyModel(this.environment);
  }

  /**
   * Perceive the environment and update beliefs about prey location
   */
  perceive(): void {
    if (!this.targetAgent) return;

    const visionRange = 1;
    const perceivedPositions: Position[] = [];

    // Check all positions within vision range
    for (let i = -visionRange; i <= visionRange; i++) {
      for (let j = -visionRange; j <= visionRange; j++) {
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
      if (pos[0] === this.targetAgent.position[0] && pos[1] === this.targetAgent.position[1]) {
        this.updateBelief(this.targetAgent.position);
        preyFound = true;
        break;
      }
    }

    if (!preyFound) {
      this.updateBelief(null);
    }
  }

  /**
   * Update belief about prey's location based on observation and model for prey behavior
   */
  private updateBelief(preyPosition: Position | null): void {
    if (preyPosition) {
      // If prey is observed, update model and set probabilities based on model
      // Reset belief grid to 0
      for (let i = 0; i < this.environment.size; i++) {
        this.preyBelief[i].fill(0);
      }

      // Update the generative model with the observed prey position
      this.preyModel.update(preyPosition);

      // Get movement probabilities from the model
      const movementProbs = this.preyModel.getMovementProbabilities();

      // Set probabilities in positions adjacent to prey based on model
      movementProbs.forEach((probability, moveKey) => {
        const move: Position = this.arrayFromString(moveKey);
        const possiblePreyPos = this.environment.normalizePosition([
          preyPosition[0] + move[0],
          preyPosition[1] + move[1]
        ]);
        this.preyBelief[possiblePreyPos[0]][possiblePreyPos[1]] = probability;
      });
    } else {
      // If prey is not observed, update beliefs with decay factor
      let sum = 0;
      for (let i = 0; i < this.environment.size; i++) {
        for (let j = 0; j < this.environment.size; j++) {
          this.preyBelief[i][j] *= 0.99;
          sum += this.preyBelief[i][j];
        }
      }

      // Normalize to ensure probabilities sum to 1
      if (sum > 0) {
        for (let i = 0; i < this.environment.size; i++) {
          for (let j = 0; j < this.environment.size; j++) {
            this.preyBelief[i][j] /= sum;
          }
        }
      }
    }
  }

  /**
   * Choose and execute action based on active inference principles
   */
  act(): void {
    let bestMove: Position | null = null;
    let highestBelief = -1;

    // Consider all possible moves using free energy principle
    // Move to the position with highest belief of finding prey
    for (let dirIndex = 0; dirIndex < DIRECTION_VECTORS.length; dirIndex++) {
      const dir = DIRECTION_VECTORS[dirIndex];
      const newPos = this.environment.normalizePosition([
        this.position[0] + dir[0],
        this.position[1] + dir[1]
      ]);

      // Check belief value at the new position
      const beliefAtNewPos = this.preyBelief[newPos[0]][newPos[1]];
      if (beliefAtNewPos > highestBelief) {
        highestBelief = beliefAtNewPos;
        bestMove = [...dir]; // Create a mutable copy
      }
    }

    // Execute the best move if found
    if (bestMove) {
      this.position = this.environment.normalizePosition([
        this.position[0] + bestMove[0],
        this.position[1] + bestMove[1]
      ]);
    }
  }

  /**
  * Since we use stringified arrays as Map keys for the prey model
  * we use this to convert them back to arrays (not safely)
  */
  arrayFromString(str: string): Position {
    return str.split(',').map(Number) as Position;
  }

  /**
   * Reset the predator's state
   */
  reset(): void {
    this.preyModel.reset();
    // Re-initialize beliefs to a new array with uniform distribution
    // This ensures the array is the correct size even if grid size has changed
    this.preyBelief = Array(this.environment.size).fill(0).map(() =>
      Array(this.environment.size).fill(1 / (this.environment.size * this.environment.size))
    );
  }
}
