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

  constructor(
    canvasContainer: HTMLElement,
    width: number = 500,
    height: number = 500
  ) {
    this.app = new PIXI.Application({
      width,
      height,
      antialias: true,
      backgroundColor: 0xf0f0f0
    });

    canvasContainer.appendChild(this.app.view as unknown as Node);

    this.gridSize = 32;
    this.cellSize = width / this.gridSize;

    this.beliefHeatmap = new PIXI.Graphics();
    this.visionHeatmap = new PIXI.Graphics();
    this.gridLines = new PIXI.Graphics();

    this.app.stage.addChild(this.beliefHeatmap);
    this.app.stage.addChild(this.visionHeatmap);
    this.app.stage.addChild(this.gridLines);

    this.drawGrid();
  }

  private drawGrid(): void {
    this.gridLines.clear();
    this.gridLines.lineStyle(1, 0xcccccc);

    for (let i = 0; i <= this.gridSize; i++) {
      this.gridLines.moveTo(i * this.cellSize, 0);
      this.gridLines.lineTo(i * this.cellSize, this.app.screen.height);
    }

    for (let i = 0; i <= this.gridSize; i++) {
      this.gridLines.moveTo(0, i * this.cellSize);
      this.gridLines.lineTo(this.app.screen.width, i * this.cellSize);
    }
  }

  private createAgentSprite(agentType: string): PIXI.Graphics {
    const sprite = new PIXI.Graphics();

    if (agentType === 'predator') {
      const catText = new PIXI.Text('🐱', {
        fontSize: this.cellSize * 0.8,
        align: 'center'
      });
      catText.anchor.set(0.5, 0.5);
      sprite.addChild(catText);
    } else if (agentType === 'prey') {
      const mouseText = new PIXI.Text('🐭', {
        fontSize: this.cellSize * 0.8,
        align: 'center'
      });
      mouseText.anchor.set(0.5, 0.5);
      sprite.addChild(mouseText);
    } else {
      sprite.beginFill(0x0000ff);
      sprite.drawCircle(0, 0, this.cellSize * 0.3);
      sprite.endFill();
    }

    this.app.stage.addChild(sprite);
    return sprite;
  }

  update(state: SimulationState): void {
    if (this.gridSize !== state.environment.size) {
      this.gridSize = state.environment.size;
      this.cellSize = this.app.screen.width / this.gridSize;
      this.drawGrid();
    }

    this.beliefHeatmap.clear();
    this.visionHeatmap.clear();

    if (state.predatorBelief) {
      this.updateBeliefHeatmap(state.predatorBelief as number[][]);
    }
    if (state.predatorVision) {
      this.updatePredatorVision(state.predatorVision);
    }

    for (const agent of state.agents) {
      let sprite = this.sprites.get(agent.id);

      if (!sprite) {
        sprite = this.createAgentSprite(agent.type);
        this.sprites.set(agent.id, sprite);
      }

      sprite.x =
        agent.position[1] * this.cellSize + this.cellSize / 2;
      sprite.y =
        agent.position[0] * this.cellSize + this.cellSize / 2;
    }

    const currentAgentIds = new Set(state.agents.map(a => a.id));
    for (const [id, sprite] of this.sprites.entries()) {
      if (!currentAgentIds.has(id)) {
        this.app.stage.removeChild(sprite);
        this.sprites.delete(id);
      }
    }
  }

  private updateBeliefHeatmap(beliefMatrix: number[][]): void {
    this.beliefHeatmap.clear();

    for (let i = 0; i < beliefMatrix.length; i++) {
      for (let j = 0; j < beliefMatrix[i].length; j++) {
        const normalizedBelief = beliefMatrix[i][j];
        const alpha = Math.min(normalizedBelief * 0.7, 0.7);

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

  private updatePredatorVision(viewedPositions: Position[]): void {
    this.visionHeatmap.clear();
    const alpha = 0.5;

    const visibleCells = new Set(
      viewedPositions.map(pos => `${pos[0]},${pos[1]}`)
    );

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

  resize(width: number, height: number): void {
    this.app.renderer.resize(width, height);
    this.cellSize = width / this.gridSize;
    this.drawGrid();
  }
}
