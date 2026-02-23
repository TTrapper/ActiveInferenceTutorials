import * as tf from '@tensorflow/tfjs';
import { Agent, GridRenderable, Position } from '../core/types';
import { GridWorld } from '../environments/gridworld';
import { PreyGenerativeModel } from './predator';

const GRID_SIZE = 32;
const NUM_TOKENS = GRID_SIZE * GRID_SIZE; // 1024
const NUM_CATEGORIES = 4; // 0=empty, 1=prey, 2=predator, 3=wall
const HIDDEN_DIM = 16;
const FFN_DIM = 64;
const LEARNING_RATE = 1e-3;
const BUFFER_SIZE = 4096;
const BATCH_SIZE = 16;

/**
 * Generate sinusoidal position encodings
 */
function generateSinusoidalPositionEncoding(length: number, dimension: number): tf.Tensor2D {
  return tf.tidy(() => {
    const position = tf.range(0, length).reshape([-1, 1]);
    const i = tf.range(0, dimension, 2).reshape([1, -1]);
    const divTerm = tf.exp(i.mul(-Math.log(10000.0) / dimension));
    const sin = tf.sin(position.mul(divTerm));
    const cos = tf.cos(position.mul(divTerm));
    return tf.concat([sin, cos], 1);
  });
}

/**
 * Xavier/Glorot uniform initializer for a given shape.
 * Returns values in [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out))
 */
function glorotUniform(shape: number[]): tf.Tensor {
  const fanIn = shape.length > 1 ? shape[0] : shape[0];
  const fanOut = shape.length > 1 ? shape[1] : shape[0];
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return tf.randomUniform(shape, -limit, limit);
}

/**
 * Transformer-based world model that learns prey movement patterns.
 *
 * Treats the 32x32 grid as 1024 tokens (one per cell), each with a
 * categorical value (empty/prey/predator). Two transformer layers
 * with self-attention process all tokens, and each token outputs a
 * single logit. Softmax over all 1024 logits gives a probability
 * distribution over where the prey moves next.
 *
 * Trained with replay buffer, cross-entropy loss, and Adam.
 */
export class TransformerWorldModel implements PreyGenerativeModel {
  private environment: GridWorld;
  private stateItems: GridRenderable[];

  // Tracking for online learning
  private lastGridState: number[] | null = null;
  private lastPreyPosition: Position | null = null;

  // Replay buffer for batch training
  private replayBuffer: { gridState: number[], targetIndex: number }[] = [];
  private bufferIndex = 0;

  // Embedding parameters
  private contentEmbed!: tf.Variable; // [3, 16]
  private posEncodingX!: tf.Tensor2D; // [32, 16]
  private posEncodingY!: tf.Tensor2D; // [32, 16]

  // Attention parameters
  private Wq!: tf.Variable; // [16, 16]
  private Wk!: tf.Variable;
  private Wv!: tf.Variable;
  private bq!: tf.Variable; // [16]
  private bk!: tf.Variable;
  private bv!: tf.Variable;
  private Wo!: tf.Variable; // [16, 16]
  private bo!: tf.Variable; // [16]
  private ln1Gamma!: tf.Variable; // [16]
  private ln1Beta!: tf.Variable;  // [16]

  // Layer 1 FFN parameters
  private W1!: tf.Variable; // [16, 64]
  private b1!: tf.Variable; // [64]
  private W2!: tf.Variable; // [64, 16]
  private b2!: tf.Variable; // [16]
  private ln2Gamma!: tf.Variable; // [16]
  private ln2Beta!: tf.Variable;  // [16]

  // Layer 2 attention parameters
  private Wq2!: tf.Variable; // [16, 16]
  private Wk2!: tf.Variable;
  private Wv2!: tf.Variable;
  private bq2!: tf.Variable; // [16]
  private bk2!: tf.Variable;
  private bv2!: tf.Variable;
  private Wo2!: tf.Variable; // [16, 16]
  private bo2!: tf.Variable; // [16]
  private ln3Gamma!: tf.Variable; // [16]
  private ln3Beta!: tf.Variable;  // [16]

  // Layer 2 FFN parameters
  private W3!: tf.Variable; // [16, 64]
  private b3!: tf.Variable; // [64]
  private W4!: tf.Variable; // [64, 16]
  private b4!: tf.Variable; // [16]
  private ln4Gamma!: tf.Variable; // [16]
  private ln4Beta!: tf.Variable;  // [16]

  // Output head
  private Wout!: tf.Variable; // [16, 1]
  private bout!: tf.Variable; // [1]

  private optimizer!: tf.AdamOptimizer;
  private allVariables: tf.Variable[] = [];

  constructor(environment: GridWorld, stateItems: GridRenderable[]) {
    this.environment = environment;
    this.stateItems = stateItems;
    this.initializeWeights();
    this.optimizer = tf.train.adam(LEARNING_RATE);

    // Warm-up forward pass to compile WebGL shaders
    tf.tidy(() => {
      const dummy = new Array(NUM_TOKENS).fill(0);
      this.forward(dummy);
    });
  }

  /**
   * Initialize all model parameters with Xavier/Glorot initialization
   */
  private initializeWeights(): void {
    this.allVariables = [];

    // Content embeddings (learned)
    this.contentEmbed = tf.variable(
      glorotUniform([NUM_CATEGORIES, HIDDEN_DIM]), true, 'contentEmbed'
    );

    // Position encodings (fixed sinusoidal, each gets half dimension)
    this.posEncodingX = generateSinusoidalPositionEncoding(GRID_SIZE, HIDDEN_DIM / 2);
    this.posEncodingY = generateSinusoidalPositionEncoding(GRID_SIZE, HIDDEN_DIM / 2);

    // Attention
    this.Wq = tf.variable(
      glorotUniform([HIDDEN_DIM, HIDDEN_DIM]), true, 'Wq'
    );
    this.Wk = tf.variable(
      glorotUniform([HIDDEN_DIM, HIDDEN_DIM]), true, 'Wk'
    );
    this.Wv = tf.variable(
      glorotUniform([HIDDEN_DIM, HIDDEN_DIM]), true, 'Wv'
    );
    this.bq = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'bq');
    this.bk = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'bk');
    this.bv = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'bv');
    this.Wo = tf.variable(
      glorotUniform([HIDDEN_DIM, HIDDEN_DIM]), true, 'Wo'
    );
    this.bo = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'bo');
    this.ln1Gamma = tf.variable(tf.ones([HIDDEN_DIM]), true, 'ln1Gamma');
    this.ln1Beta = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'ln1Beta');

    // Layer 1 FFN
    this.W1 = tf.variable(
      glorotUniform([HIDDEN_DIM, FFN_DIM]), true, 'W1'
    );
    this.b1 = tf.variable(tf.zeros([FFN_DIM]), true, 'b1');
    this.W2 = tf.variable(
      glorotUniform([FFN_DIM, HIDDEN_DIM]), true, 'W2'
    );
    this.b2 = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'b2');
    this.ln2Gamma = tf.variable(tf.ones([HIDDEN_DIM]), true, 'ln2Gamma');
    this.ln2Beta = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'ln2Beta');

    // Layer 2 Attention
    this.Wq2 = tf.variable(
      glorotUniform([HIDDEN_DIM, HIDDEN_DIM]), true, 'Wq2'
    );
    this.Wk2 = tf.variable(
      glorotUniform([HIDDEN_DIM, HIDDEN_DIM]), true, 'Wk2'
    );
    this.Wv2 = tf.variable(
      glorotUniform([HIDDEN_DIM, HIDDEN_DIM]), true, 'Wv2'
    );
    this.bq2 = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'bq2');
    this.bk2 = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'bk2');
    this.bv2 = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'bv2');
    this.Wo2 = tf.variable(
      glorotUniform([HIDDEN_DIM, HIDDEN_DIM]), true, 'Wo2'
    );
    this.bo2 = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'bo2');
    this.ln3Gamma = tf.variable(tf.ones([HIDDEN_DIM]), true, 'ln3Gamma');
    this.ln3Beta = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'ln3Beta');

    // Layer 2 FFN
    this.W3 = tf.variable(
      glorotUniform([HIDDEN_DIM, FFN_DIM]), true, 'W3'
    );
    this.b3 = tf.variable(tf.zeros([FFN_DIM]), true, 'b3');
    this.W4 = tf.variable(
      glorotUniform([FFN_DIM, HIDDEN_DIM]), true, 'W4'
    );
    this.b4 = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'b4');
    this.ln4Gamma = tf.variable(tf.ones([HIDDEN_DIM]), true, 'ln4Gamma');
    this.ln4Beta = tf.variable(tf.zeros([HIDDEN_DIM]), true, 'ln4Beta');

    // Output head
    this.Wout = tf.variable(
      glorotUniform([HIDDEN_DIM, 1]), true, 'Wout'
    );
    this.bout = tf.variable(tf.zeros([1]), true, 'bout');

    this.allVariables = [
      this.contentEmbed,
      this.Wq, this.Wk, this.Wv, this.bq, this.bk, this.bv,
      this.Wo, this.bo, this.ln1Gamma, this.ln1Beta,
      this.W1, this.b1, this.W2, this.b2, this.ln2Gamma, this.ln2Beta,
      this.Wq2, this.Wk2, this.Wv2, this.bq2, this.bk2, this.bv2,
      this.Wo2, this.bo2, this.ln3Gamma, this.ln3Beta,
      this.W3, this.b3, this.W4, this.b4, this.ln4Gamma, this.ln4Beta,
      this.Wout, this.bout
    ];
  }

  /**
   * Layer normalization over the last dimension
   */
  private layerNorm(
    x: tf.Tensor2D, gamma: tf.Variable, beta: tf.Variable
  ): tf.Tensor2D {
    const mean = x.mean(-1, true);
    const variance = x.sub(mean).square().mean(-1, true);
    const normalized = x.sub(mean).div(variance.add(1e-5).sqrt());
    return normalized.mul(gamma).add(beta) as tf.Tensor2D;
  }

  /**
   * Forward pass through the transformer.
   * @param gridState Flat array of 1024 categorical values (0/1/2)
   * @returns 1024 logits (before softmax)
   */
  forward(gridState: number[]): tf.Tensor1D {
    // Build token embeddings: content + pos (fixed)
    const indices = tf.tensor1d(gridState, 'int32');
    const contentVecs = tf.gather(this.contentEmbed, indices); // [1024, 16]

    // Build x and y indices for each token
    const xIndices = tf.tensor1d(
      Array.from({ length: NUM_TOKENS }, (_, i) => Math.floor(i / GRID_SIZE)),
      'int32'
    );
    const yIndices = tf.tensor1d(
      Array.from({ length: NUM_TOKENS }, (_, i) => i % GRID_SIZE),
      'int32'
    );
    const xPosVecs = tf.gather(this.posEncodingX, xIndices); // [1024, 8]
    const yPosVecs = tf.gather(this.posEncodingY, yIndices); // [1024, 8]

    // Concatenate X and Y encodings, then add to content
    const posEnc = tf.concat([xPosVecs, yPosVecs], 1); // [1024, 16]
    let x = contentVecs.add(posEnc) as tf.Tensor2D;

    const scale = Math.sqrt(HIDDEN_DIM);

    // Layer 1: Self-attention with pre-norm
    const normed1 = this.layerNorm(x, this.ln1Gamma, this.ln1Beta);
    const Q1 = normed1.matMul(this.Wq).add(this.bq); // [1024, 16]
    const K1 = normed1.matMul(this.Wk).add(this.bk);
    const V1 = normed1.matMul(this.Wv).add(this.bv);
    const scores1 = Q1.matMul(K1.transpose()).div(scale);
    const attnW1 = scores1.softmax(-1);
    const attnOut1 = attnW1.matMul(V1);
    const proj1 = attnOut1.matMul(this.Wo).add(this.bo);
    x = x.add(proj1) as tf.Tensor2D;

    // Layer 1: FFN with pre-norm
    const normed2 = this.layerNorm(x, this.ln2Gamma, this.ln2Beta);
    const ffn1 = normed2.matMul(this.W1).add(this.b1).relu();
    const ffnOut1 = ffn1.matMul(this.W2).add(this.b2);
    x = x.add(ffnOut1) as tf.Tensor2D;

    // Layer 2: Self-attention with pre-norm
    const normed3 = this.layerNorm(x, this.ln3Gamma, this.ln3Beta);
    const Q2 = normed3.matMul(this.Wq2).add(this.bq2);
    const K2 = normed3.matMul(this.Wk2).add(this.bk2);
    const V2 = normed3.matMul(this.Wv2).add(this.bv2);
    const scores2 = Q2.matMul(K2.transpose()).div(scale);
    const attnW2 = scores2.softmax(-1);
    const attnOut2 = attnW2.matMul(V2);
    const proj2 = attnOut2.matMul(this.Wo2).add(this.bo2);
    x = x.add(proj2) as tf.Tensor2D;

    // Layer 2: FFN with pre-norm
    const normed4 = this.layerNorm(x, this.ln4Gamma, this.ln4Beta);
    const ffn2 = normed4.matMul(this.W3).add(this.b3).relu();
    const ffnOut2 = ffn2.matMul(this.W4).add(this.b4);
    x = x.add(ffnOut2) as tf.Tensor2D;

    // Output head: [1024, 16] -> [1024, 1] -> [1024]
    const logits = x.matMul(this.Wout).add(this.bout).squeeze([1]);
    return logits as tf.Tensor1D;
  }

  /**
   * Add a transition to the replay buffer (circular)
   */
  private addToBuffer(gridState: number[], targetIndex: number): void {
    if (this.replayBuffer.length < BUFFER_SIZE) {
      this.replayBuffer.push({ gridState, targetIndex });
    } else {
      this.replayBuffer[this.bufferIndex] = { gridState, targetIndex };
    }
    this.bufferIndex = (this.bufferIndex + 1) % BUFFER_SIZE;
  }

  /**
   * Sample a random batch from the replay buffer
   */
  private sampleBatch(
    size: number
  ): { gridState: number[], targetIndex: number }[] {
    const batch: { gridState: number[], targetIndex: number }[] = [];
    for (let i = 0; i < size; i++) {
      const idx = Math.floor(Math.random() * this.replayBuffer.length);
      batch.push(this.replayBuffer[idx]);
    }
    return batch;
  }

  /**
   * Train the model on a batch sampled from the replay buffer
   */
  private trainStep(): void {
    if (this.replayBuffer.length < BATCH_SIZE) return;

    const batch = this.sampleBatch(BATCH_SIZE);

    this.optimizer.minimize(() => {
      let totalLoss: tf.Scalar = tf.scalar(0);
      for (const example of batch) {
        const target = tf.oneHot(
          tf.tensor1d([example.targetIndex], 'int32'), NUM_TOKENS
        );
        const logits = this.forward(example.gridState);
        const logitsReshaped = logits.reshape([1, NUM_TOKENS]);
        const loss = tf.losses.softmaxCrossEntropy(
          target, logitsReshaped
        );
        totalLoss = totalLoss.add(loss) as tf.Scalar;
      }
      return totalLoss.div(tf.scalar(batch.length)) as tf.Scalar;
    }, true, this.allVariables);
  }

  /**
   * Build the flat grid state array from current environment state
   */
  private buildGridState(): number[] {
    const state = new Array(NUM_TOKENS).fill(0);

    // Add walls from environment
    const walls = this.environment.getWalls();
    for (const wallPos of walls) {
      const idx = wallPos[0] * GRID_SIZE + wallPos[1];
      state[idx] = 3; // wall
    }

    for (const item of this.stateItems) {
      const [x, y] = item.position;
      const idx = x * GRID_SIZE + y;
      if (item.asciiSymbol === 'r') {
        state[idx] = 1; // prey
      } else if (item.asciiSymbol === 'P') {
        state[idx] = 2; // predator
      }
    }
    return state;
  }

  /**
   * Update the model based on observed prey position.
   * Called each step by the predator agent.
   */
  update(prey: Agent | null): void {
    if (prey === null) {
      this.lastPreyPosition = null;
      this.lastGridState = null;
      return;
    }

    const currentGridState = this.buildGridState();

    if (this.lastGridState !== null && this.lastPreyPosition !== null) {
      const targetIndex =
        prey.position[0] * GRID_SIZE + prey.position[1];
      this.addToBuffer(this.lastGridState, targetIndex);
      this.trainStep();
    }

    this.lastGridState = currentGridState;
    this.lastPreyPosition = [...prey.position];
  }

  /**
   * Get movement probabilities as a direction map.
   * Converts the position distribution to relative directions from
   * the prey's current position.
   */
  getMovementProbabilities(): Map<string, number> {
    const probs = new Map<string, number>();

    const preyItem = this.stateItems.find(
      item => item.asciiSymbol === 'r'
    );
    if (!preyItem) return probs;

    const positionGrid = this.getPositionGrid();
    const [preyX, preyY] = preyItem.position;

    // Convert position probabilities to direction probabilities
    for (let x = 0; x < GRID_SIZE; x++) {
      for (let y = 0; y < GRID_SIZE; y++) {
        const p = positionGrid[x][y];
        if (p > 1e-4) {
          const dx = x - preyX;
          const dy = y - preyY;
          const key = `${dx},${dy}`;
          probs.set(key, (probs.get(key) ?? 0) + p);
        }
      }
    }

    return probs;
  }

  /**
   * Get the position probability grid (32x32) directly from the model.
   * Used by the renderer to display the heatmap without direction
   * conversion.
   */
  getPositionGrid(): number[][] {
    const gridState = this.buildGridState();
    const grid: number[][] = tf.tidy(() => {
      const logits = this.forward(gridState);
      const probsTensor = logits.softmax();
      const probs2D = probsTensor.reshape([GRID_SIZE, GRID_SIZE]);
      return probs2D.arraySync() as number[][];
    });
    return grid;
  }

  /**
   * Reset the model: re-initialize all weights and clear state
   */
  reset(): void {
    this.dispose();
    this.initializeWeights();
    this.optimizer = tf.train.adam(LEARNING_RATE);
    this.lastGridState = null;
    this.lastPreyPosition = null;
    this.replayBuffer = [];
    this.bufferIndex = 0;

    // Warm-up forward pass
    tf.tidy(() => {
      const dummy = new Array(NUM_TOKENS).fill(0);
      this.forward(dummy);
    });
  }

  /**
   * Dispose all TensorFlow variables and optimizer state
   */
  dispose(): void {
    for (const v of this.allVariables) {
      v.dispose();
    }
    this.allVariables = [];
    this.posEncodingX.dispose();
    this.posEncodingY.dispose();
    this.optimizer.dispose();
  }
}
