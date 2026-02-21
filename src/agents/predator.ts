import { Agent, DIRECTION_VECTORS, Position } from '../core/types';
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
 * Basic prey model with uniform movement probabilities (Lesson 2)
 */
export class UniformPreyModel implements PreyGenerativeModel {
  constructor(private environment: GridWorld) {}

  update(prey: Agent | null): void {
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

  update(prey: Agent | null): void {
    // If the prey wasn't seen, set lastPreyPosition to null because we only update
    // on consequtive moves.
    // TODO We should apply a soft update when lastPreyPosition = null, based on belief about where the prey *could* have come from
    if (prey === null) {
      this.lastPreyPosition = null;
      return;
    }
    if (this.lastPreyPosition !== null) {
      const directionMoved = this.computeDirectionMoved(this.lastPreyPosition, prey.position);
      const currentCount = this.movementCounts.get(directionMoved.toString()) ?? 0;
      this.movementCounts.set(directionMoved.toString(), currentCount + 1);
    }
    this.lastPreyPosition = [...prey.position];
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
 * Bayesian prey model that learns world states (Lesson 4)
 *
 * Like Lesson 3 except instead of modeling relative movements of Prey,
 * now we model each possible position of the prey independently. This will results
 * in a larger lookup table and much slower learning, but doesn't assume the prey
 * behaves the same in all situations. Opens up the ability to add more state to the
 * world, like the predator's own position, walls, food, etc.
 *
 */
export class BayesianWorldModel implements PreyGenerativeModel {
  private lastState: string | null = null;
  private lastPreyPosition: Position | null = null;
  private transitionCounts: Map<string, Map<string, number>> = new Map();

  constructor(private environment: GridWorld) {
    this.reset();
  }

  update(prey: Agent | null): void {
    if (prey === null) {
      this.lastPreyPosition = null;
      return;
    }
    const currentState = this.environment.gridToString([prey]);
    if (this.lastState !== null && this.lastPreyPosition !== null) {
      const directionMoved = this.computeDirectionMoved(this.lastPreyPosition, prey.position);
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
/**
 * Available generative model types for the predator
 */
export enum GenerativeModelType {
  UNIFORM = 'uniform',
  BAYESIAN_PREY = 'bayesian_prey',
  BAYESIAN_WORLD = 'bayesian_world'
}

export class ActiveInferencePredator implements Agent {
  id: string;
  position: Position;
  asciiSymbol = 'P';
  environment: GridWorld;
  preyBelief: number[][];
  preyModel!: PreyGenerativeModel;
  targetAgent: Agent | null = null;
  visionRange: number;

  constructor(id: string, position: Position, environment: GridWorld, modelType: GenerativeModelType = GenerativeModelType.UNIFORM) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;
    this.visionRange = environment.size;

    // Initialize belief about prey's position to uniform distribution
    this.preyBelief = Array(environment.size).fill(0).map(() =>
      Array(environment.size).fill(1 / (environment.size * environment.size))
    );

    this.setGenerativeModel(environment, modelType);
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
  setGenerativeModel(environment: GridWorld, modelType: GenerativeModelType = GenerativeModelType.UNIFORM): void {
    switch (modelType) {
      case GenerativeModelType.UNIFORM:
        this.preyModel = new UniformPreyModel(environment); break;
      case GenerativeModelType.BAYESIAN_PREY:
        this.preyModel = new BayesianPreyModel(environment); break;
      case GenerativeModelType.BAYESIAN_WORLD:
        this.preyModel = new BayesianWorldModel(environment); break;
    }
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
    if (!this.targetAgent) return;

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
      if (pos[0] === this.targetAgent.position[0] && pos[1] === this.targetAgent.position[1]) {
        if (doUpdateBelief) {
          this.updateBelief(this.targetAgent.position, perceivedPositions);
        }
        preyFound = true;
        break;
      }
    }

    if (!preyFound) {
      this.updateBelief(null, perceivedPositions);
    }

    return perceivedPositions
  }

  /**
   * Update belief about prey's NEXT location based on observation and model for prey behavior
   * TODO cleanly separate belief about the CURRENT state of the world from PREDICTION about the future and ACTION
   */
  private updateBelief(preyPosition: Position | null, perceivedPositions: Array<Position>): void {
    if (preyPosition) {
      // If prey is observed, update model and set probabilities based on model
      // Reset belief grid to 0
      for (let i = 0; i < this.environment.size; i++) {
        this.preyBelief[i].fill(0);
      }

      // Update the generative model with the observed prey
      // FIXME preyModel needs to take into account the situation where we didn't see the prey in the last step
      this.preyModel.update(this.targetAgent);

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
      /* Spread the probability of the prey's next position according to the predator's belief
       * about where the prey might be and the model of the prey's movement.
       * NOTE this is a convolution over belief with the preyModel as the kernel
       */

      this.preyModel.update(null);
      // Set the spots we just saw to zero, because we didn't see the prey there
      for (let position of perceivedPositions) {
        this.preyBelief[position[0]][position[1]] = 0.0
      }
      this.normalizePreyBelief();
      // Initialize a new belief matrix
      const newPreyBelief : number[][] = Array(this.environment.size)
        .fill(0.0)
        .map(() =>
          Array(this.environment.size).fill(0)
        );
      // Do the convolution
      for (let i = 0; i < this.environment.size; i++) {
        for (let j = 0; j < this.environment.size; j++) {
          this.preyModel.getMovementProbabilities().forEach((probability, moveKey) => {
            const move: Position = this.arrayFromString(moveKey);
            const possiblePreyPos : Position = this.environment.normalizePosition(
              [i + move[0], j + move[1]]
            )
            newPreyBelief[possiblePreyPos[0]][possiblePreyPos[1]] += probability * this.preyBelief[i][j]
          });
        }
      }
      this.preyBelief = newPreyBelief;
      this.normalizePreyBelief();
    }
  }

  /**
  * Samples one element from the options array based on the weights.
  * @param options Array of actions.
  * @param weights Array of weights corresponding to each action.
  * @returns A sampled action.
  */
  sampleFromWeights<T>(options: T[], weights: number[]): T | null {
    const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
    if (totalWeight === 0) return null;
    let threshold = Math.random() * totalWeight;
    for (let i = 0; i < options.length; i++) {
      threshold -= weights[i];
      if (threshold <= 0) {
        return options[i];
      }
    }
    return options[options.length - 1] || null; // Fallback in case of numerical issues.
  }

  /**
  * Choose and execute action based on active inference principles,
  * but instead of teleporting to the sampled location, take a step
  * in its direction with integer movements.
  */
  act(): void {
    const possibleMoves: Position[] = [];
    const moveWeights: number[] = [];

    // Calculate weights for each possible move based on the belief grid.
    for (let x = 0; x < this.preyBelief.length; x++) {
      for (let y = 0; y < this.preyBelief[x].length; y++) {
        possibleMoves.push([x, y]);
        moveWeights.push(this.preyBelief[x][y]);
      }
    }

    // Sample an action according to the computed weights.
    const selectedMove = this.sampleFromWeights(possibleMoves, moveWeights);
    if (selectedMove) {
      const targetPos: Position = [selectedMove[0], selectedMove[1]];
      const currentPos: Position = this.position;

      // Compute the difference in each coordinate.
      const dx = targetPos[0] - currentPos[0];
      const dy = targetPos[1] - currentPos[1];

      // Determine a discrete step by taking the sign of the differences.
      const stepX = Math.sign(dx); // will be -1, 0, or 1
      const stepY = Math.sign(dy); // will be -1, 0, or 1

      // Update position by adding the discrete step.
      const newPos: Position = [currentPos[0] + stepX, currentPos[1] + stepY];

      // Use the environment's normalization if needed (e.g., to handle wrapping or boundaries).
      this.position = this.environment.normalizePosition(newPos);
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
   * Make preyBelief so probabilities sum to 1
   */
  normalizePreyBelief() {
    let sum = 0;
    for (let i = 0; i < this.environment.size; i++) {
      for (let j = 0; j < this.environment.size; j++){
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
    // Re-initialize beliefs to a new array with uniform distribution
    // This ensures the array is the correct size even if grid size has changed
    this.preyBelief = Array(this.environment.size).fill(0).map(() =>
      Array(this.environment.size).fill(1 / (this.environment.size * this.environment.size))
    );
  }
}
