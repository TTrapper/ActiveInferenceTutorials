import * as PIXI from 'pixi.js';
import { SimulationState, Position } from '../core/types';

export type HeatmapMode =
  | 'prey_policy'
  | 'predator_belief'
  | 'model_error';

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
  heatmapMode: HeatmapMode = 'prey_policy';

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

  setHeatmapMode(mode: HeatmapMode): void {
    this.heatmapMode = mode;
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

    // Render heatmap based on selected mode
    switch (this.heatmapMode) {
      case 'prey_policy':
        if (state.preyTrueProbs) {
          this.renderHeatmap(state.preyTrueProbs, 0x00cc00);
        }
        break;
      case 'predator_belief':
        if (state.predatorBelief) {
          this.renderHeatmap(state.predatorBelief, 0x0066ff);
        }
        break;
      case 'model_error':
        if (state.preyTrueProbs && state.predatorModelProbs) {
          const error = this.computeAbsError(
            state.preyTrueProbs, state.predatorModelProbs
          );
          this.renderHeatmap(error, 0xff3300);
        }
        break;
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
        agent.position[0] * this.cellSize + this.cellSize / 2;
      sprite.y =
        agent.position[1] * this.cellSize + this.cellSize / 2;
    }

    const currentAgentIds = new Set(state.agents.map(a => a.id));
    for (const [id, sprite] of this.sprites.entries()) {
      if (!currentAgentIds.has(id)) {
        this.app.stage.removeChild(sprite);
        this.sprites.delete(id);
      }
    }
  }

  /**
   * Render a 2D matrix as a heatmap, normalized by its max value
   */
  private renderHeatmap(matrix: number[][], color: number): void {
    let maxVal = 0;
    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        if (matrix[i][j] > maxVal) maxVal = matrix[i][j];
      }
    }
    if (maxVal === 0) return;

    for (let i = 0; i < matrix.length; i++) {
      for (let j = 0; j < matrix[i].length; j++) {
        const val = matrix[i][j];
        if (val > 0) {
          const alpha = (val / maxVal) * 0.7;
          this.beliefHeatmap.beginFill(color, alpha);
          this.beliefHeatmap.drawRect(
            i * this.cellSize,
            j * this.cellSize,
            this.cellSize,
            this.cellSize
          );
          this.beliefHeatmap.endFill();
        }
      }
    }
  }

  /**
   * Compute element-wise absolute difference between two matrices
   */
  private computeAbsError(
    a: number[][], b: number[][]
  ): number[][] {
    const rows = Math.min(a.length, b.length);
    const result: number[][] = [];
    for (let i = 0; i < rows; i++) {
      const cols = Math.min(a[i].length, b[i].length);
      const row: number[] = [];
      for (let j = 0; j < cols; j++) {
        row.push(Math.abs(a[i][j] - b[i][j]));
      }
      result.push(row);
    }
    return result;
  }

  private updatePredatorVision(viewedPositions: Position[]): void {
    this.visionHeatmap.clear();
    const alpha = 0.5;

    const visibleCells = new Set(
      viewedPositions.map(pos => `${pos[0]},${pos[1]}`)
    );

    for (let y = 0; y < this.gridSize; y++) {
      for (let x = 0; x < this.gridSize; x++) {
        if (!visibleCells.has(`${x},${y}`)) {
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
