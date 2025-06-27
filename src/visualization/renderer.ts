import * as PIXI from 'pixi.js';
import { SimulationState, Position } from '../core/types';

/**
 * Renderer for the simulation using PixiJS
 */
export class PixiRenderer {
  app: PIXI.Application;
  gridSize: number;
  cellSize: number;
  sprites: Map<string, PIXI.Graphics> = new Map();
  gridLines: PIXI.Graphics;
  beliefHeatmap: PIXI.Graphics;
  visionHeatmap: PIXI.Graphics;

  constructor(canvasContainer: HTMLElement, width: number = 500, height: number = 500) {
    // Initialize PIXI Application
    this.app = new PIXI.Application({
      width,
      height,
      antialias: true,
      backgroundColor: 0xf0f0f0
    });
    
    canvasContainer.appendChild(this.app.view as unknown as Node);

    this.gridSize = 32; // Fixed at 32x32 for predator-prey
    this.cellSize = width / this.gridSize;
    
    // Create graphics containers
    this.beliefHeatmap = new PIXI.Graphics();
    this.visionHeatmap = new PIXI.Graphics();
    this.gridLines = new PIXI.Graphics();
    
    this.app.stage.addChild(this.beliefHeatmap);
    this.app.stage.addChild(this.visionHeatmap);
    this.app.stage.addChild(this.gridLines);
    
    // Draw initial grid
    this.drawGrid();
  }

  /**
   * Draw the grid lines
   */
  private drawGrid(): void {
    this.gridLines.clear();
    this.gridLines.lineStyle(1, 0xcccccc);
    
    // Draw vertical lines
    for (let i = 0; i <= this.gridSize; i++) {
      this.gridLines.moveTo(i * this.cellSize, 0);
      this.gridLines.lineTo(i * this.cellSize, this.app.screen.height);
    }
    
    // Draw horizontal lines
    for (let i = 0; i <= this.gridSize; i++) {
      this.gridLines.moveTo(0, i * this.cellSize);
      this.gridLines.lineTo(this.app.screen.width, i * this.cellSize);
    }
  }

  /**
   * Create a sprite for an agent
   */
  private createAgentSprite(agentType: string): PIXI.Graphics {
    const sprite = new PIXI.Graphics();
    
    // Different graphics based on agent type
    if (agentType === 'predator') {
      // Create cat emoji text
      const catText = new PIXI.Text('🐱', {
        fontSize: this.cellSize * 0.8,
        align: 'center'
      });
      catText.anchor.set(0.5, 0.5);
      sprite.addChild(catText);
    } else if (agentType === 'prey') {
      // Create mouse emoji text
      const mouseText = new PIXI.Text('🐭', {
        fontSize: this.cellSize * 0.8,
        align: 'center'
      });
      mouseText.anchor.set(0.5, 0.5);
      sprite.addChild(mouseText);
    } else if (agentType === 'state_machine') {
      sprite.beginFill(0x9933ff); // Purple for state machine
      sprite.drawRect(-this.cellSize * 0.25, -this.cellSize * 0.25, 
                       this.cellSize * 0.5, this.cellSize * 0.5);
      sprite.endFill();
    } else {
      // Default appearance
      sprite.beginFill(0x0000ff);
      sprite.drawCircle(0, 0, this.cellSize * 0.3);
      sprite.endFill();
    }
    
    this.app.stage.addChild(sprite);
    
    return sprite;
  }

  /**
   * Update visualization based on current simulation state
   */
  update(state: SimulationState): void {
    // Update grid size if needed
    if (this.gridSize !== state.environment.size) {
      this.gridSize = state.environment.size;
      this.cellSize = this.app.screen.width / this.gridSize;
      this.drawGrid();
    }
    
    // Clear previous graphics
    this.beliefHeatmap.clear();
    this.visionHeatmap.clear();
    
    // Update belief heatmap based on simulation type
    if (state.predatorBelief) {
      // For predator-prey simulation: show predator's belief about prey
      this.updateBeliefHeatmap(state.predatorBelief as number[][]);
    } else if (state.transitionMatrix) {
      // For state machine: show transition probabilities from current state
      this.updateTransitionMatrixVisualization(state);
    }
    if (state.predatorVision) {
      // For predator-prey simulation: show what cells the predator can see
      this.updatePredatorVision(state.predatorVision);
    }
    
    // Update or create agent sprites
    for (const agent of state.agents) {
      let sprite = this.sprites.get(agent.id);
      
      if (!sprite) {
        // Create new sprite if it doesn't exist
        sprite = this.createAgentSprite(agent.type);
        this.sprites.set(agent.id, sprite);
      }
      
      // Update sprite position
      sprite.x = agent.position[1] * this.cellSize + this.cellSize / 2;
      sprite.y = agent.position[0] * this.cellSize + this.cellSize / 2;
    }
    
    // Remove sprites for agents that no longer exist
    const currentAgentIds = new Set(state.agents.map(a => a.id));
    for (const [id, sprite] of this.sprites.entries()) {
      if (!currentAgentIds.has(id)) {
        this.app.stage.removeChild(sprite);
        this.sprites.delete(id);
      }
    }
  }

  /**
   * Visualize the transition matrix from the current state
   */
  private updateTransitionMatrixVisualization(state: SimulationState): void {
    if (!state.transitionMatrix || state.agents.length === 0) return;
    
    const matrix = state.transitionMatrix as number[][];
    const agent = state.agents[0]; // We only have one agent in the state machine simulation
    
    // Convert agent position to state index
    const gridSize = state.environment.size;
    const stateIndex = agent.position[0] * gridSize + agent.position[1];
    
    // Get transition probabilities from current state
    const transitionProbabilities = matrix[stateIndex];
    
    // Find max probability for normalization
    const maxProb = Math.max(...transitionProbabilities);
    
    // Draw cells with opacity based on transition probability
    for (let i = 0; i < gridSize; i++) {
      for (let j = 0; j < gridSize; j++) {
        const targetStateIndex = i * gridSize + j;
        const probability = transitionProbabilities[targetStateIndex];
        
        if (maxProb > 0 && probability > 0) {
          const normalizedProb = probability / maxProb;
          const alpha = Math.min(normalizedProb * 0.7, 0.7); // Cap at 0.7 opacity
          
          this.beliefHeatmap.beginFill(0x9933ff, alpha); // Purple for state machine
          this.beliefHeatmap.drawRect(
            j * this.cellSize, 
            i * this.cellSize, 
            this.cellSize, 
            this.cellSize
          );
          this.beliefHeatmap.endFill();
        }
      }
    }
  }

  /**
   * Draw heatmap of predator's belief about prey location
   */
  private updateBeliefHeatmap(beliefMatrix: number[][]): void {
    this.beliefHeatmap.clear();

    // Draw cells with opacity based on belief strength
    for (let i = 0; i < beliefMatrix.length; i++) {
      for (let j = 0; j < beliefMatrix[i].length; j++) {
        const normalizedBelief = beliefMatrix[i][j];
        const alpha = Math.min(normalizedBelief * 0.7, 0.7); // Cap at 0.7 opacity

        this.beliefHeatmap.beginFill(0x0000ff, alpha);
        this.beliefHeatmap.drawRect(
          j * this.cellSize, 
          i * this.cellSize, 
          this.cellSize, 
          this.cellSize
        );
        this.beliefHeatmap.endFill();
      }
    }
  }

  /**
   * Black out cells the predator can't see
   */
  private updatePredatorVision(viewedPositions: Position[]): void {
    this.visionHeatmap.clear();
    const alpha = 0.5;

    // Create a set for faster lookup
    const visibleCells = new Set(viewedPositions.map(pos => `${pos[0]},${pos[1]}`));

    for (let y = 0; y < this.gridSize; y++) {
      for (let x = 0; x < this.gridSize; x++) {
        if (!visibleCells.has(`${y},${x}`)) {
          this.visionHeatmap.beginFill(0x000000, alpha);
          this.visionHeatmap.drawRect(
            x * this.cellSize,
            y * this.cellSize,
            this.cellSize,
            this.cellSize
          );
          this.visionHeatmap.endFill();
        }
      }
    }
  }

  /**
   * Resize the renderer
   */
  resize(width: number, height: number): void {
    this.app.renderer.resize(width, height);
    this.cellSize = width / this.gridSize;
    this.drawGrid();
  }
}
