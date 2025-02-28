import { Agent, DIRECTION_VECTORS, Position } from '../core/types';
import { GridWorld } from '../environments/gridworld';

/**
 * A predator agent that uses active inference principles to hunt prey
 */
export class ActiveInferencePredator implements Agent {
  id: string;
  position: Position;
  environment: GridWorld;
  preyBelief: number[][];
  targetAgent: Agent | null = null;
  
  constructor(id: string, position: Position, environment: GridWorld) {
    this.id = id;
    this.position = [...position];
    this.environment = environment;
    
    // Initialize belief about prey's position to uniform distribution
    this.preyBelief = Array(environment.size).fill(0).map(() => 
      Array(environment.size).fill(1 / (environment.size * environment.size))
    );
  }
  
  /**
   * Set the agent that this predator is hunting
   */
  setTargetAgent(agent: Agent): void {
    this.targetAgent = agent;
  }

  /**
   * Perceive the environment and update beliefs about prey location
   */
  perceive(): void {
    if (!this.targetAgent) return;
    
    const visionRange = 2;
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
   * Update belief about prey's location based on observation
   */
  private updateBelief(preyPosition: Position | null): void {
    if (preyPosition) {
      // If prey is observed, set high probability around its position
      // Reset belief grid to 0
      for (let i = 0; i < this.environment.size; i++) {
        this.preyBelief[i].fill(0);
      }

      // Set probabilities in positions adjacent to prey
      for (const dir of DIRECTION_VECTORS) {
        const possiblePreyPos = this.environment.normalizePosition([
          preyPosition[0] + dir[0],
          preyPosition[1] + dir[1]
        ]);
        this.preyBelief[possiblePreyPos[0]][possiblePreyPos[1]] = 1/8;
      }
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
    let bestMove: number[] | null = null;
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
        bestMove = dir;
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
}