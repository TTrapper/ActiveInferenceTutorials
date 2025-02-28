import * as PIXI from 'pixi.js';
import { SimulationState } from '../core/types';

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

  constructor(canvasContainer: HTMLElement, width: number = 400, height: number = 400) {
    // Initialize PIXI Application
    this.app = new PIXI.Application({
      width,
      height,
      antialias: true,
      backgroundColor: 0xf0f0f0
    });
    
    canvasContainer.appendChild(this.app.view as unknown as Node);

    this.gridSize = 8;
    this.cellSize = width / this.gridSize;
    
    // Create graphics containers
    this.beliefHeatmap = new PIXI.Graphics();
    this.gridLines = new PIXI.Graphics();
    
    this.app.stage.addChild(this.beliefHeatmap);
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
    
    // Different colors and shapes based on agent type
    if (agentType === 'predator') {
      sprite.beginFill(0xff0000);
      sprite.drawCircle(0, 0, this.cellSize * 0.3);
    } else if (agentType === 'prey') {
      sprite.beginFill(0x00cc00);
      sprite.drawCircle(0, 0, this.cellSize * 0.3);
    } else if (agentType === 'state_machine') {
      sprite.beginFill(0x9933ff); // Purple for state machine
      sprite.drawRect(-this.cellSize * 0.25, -this.cellSize * 0.25, 
                       this.cellSize * 0.5, this.cellSize * 0.5);
    } else {
      // Default appearance
      sprite.beginFill(0x0000ff);
      sprite.drawCircle(0, 0, this.cellSize * 0.3);
    }
    
    sprite.endFill();
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
    
    // Clear previous heatmap
    this.beliefHeatmap.clear();
    
    // Update belief heatmap based on simulation type
    if (state.predatorBelief) {
      // For predator-prey simulation: show predator's belief about prey
      this.updateBeliefHeatmap(state.predatorBelief as number[][]);
    } else if (state.transitionMatrix) {
      // For state machine: show transition probabilities from current state
      this.updateTransitionMatrixVisualization(state);
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
    
    // Find max belief for normalization
    let maxBelief = 0;
    for (let i = 0; i < beliefMatrix.length; i++) {
      for (let j = 0; j < beliefMatrix[i].length; j++) {
        maxBelief = Math.max(maxBelief, beliefMatrix[i][j]);
      }
    }
    
    // Draw cells with opacity based on belief strength
    for (let i = 0; i < beliefMatrix.length; i++) {
      for (let j = 0; j < beliefMatrix[i].length; j++) {
        if (maxBelief > 0) {
          const normalizedBelief = beliefMatrix[i][j] / maxBelief;
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