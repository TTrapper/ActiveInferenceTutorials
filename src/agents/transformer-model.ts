import * as tf from '@tensorflow/tfjs';
import { Agent, GridRenderable, Position } from '../core/types';
import { GridWorld } from '../environments/gridworld';
import { PreyGenerativeModel } from './predator';

const GRID_SIZE = 32;
const NUM_TOKENS = GRID_SIZE * GRID_SIZE; // 1024
const NUM_CATEGORIES = 3; // 0=empty, 1=prey, 2=predator
const HIDDEN_DIM = 16;
const FFN_DIM = 64;
const LEARNING_RATE = 1e-2;

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
 * categorical value (empty/prey/predator). A single transformer layer
 * with self-attention processes all tokens, and each token outputs a
 * single logit. Softmax over all 1024 logits gives a probability
 * distribution over where the prey moves next.
 *
 * ~4,600 parameters. Trained online with cross-entropy loss and Adam.
 */
export class TransformerWorldModel implements PreyGenerativeModel {
  private environment: GridWorld;
  private stateItems: GridRenderable[];

  // Tracking for online learning
  private lastGridState: number[] | null = null;
  private lastPreyPosition: Position | null = null;

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

  // FFN parameters
  private W1!: tf.Variable; // [16, 64]
  private b1!: tf.Variable; // [64]
  private W2!: tf.Variable; // [64, 16]
  private b2!: tf.Variable; // [16]
  private ln2Gamma!: tf.Variable; // [16]
  private ln2Beta!: tf.Variable;  // [16]

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

    // FFN
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

    // Self-attention with pre-norm
    const normed1 = this.layerNorm(x, this.ln1Gamma, this.ln1Beta);
    const Q = normed1.matMul(this.Wq).add(this.bq); // [1024, 16]
    const K = normed1.matMul(this.Wk).add(this.bk);
    const V = normed1.matMul(this.Wv).add(this.bv);

    // Scaled dot-product attention
    const scale = Math.sqrt(HIDDEN_DIM);
    const scores = Q.matMul(K.transpose()).div(scale); // [1024, 1024]
    const attnWeights = scores.softmax(-1); // [1024, 1024]
    const attnOut = attnWeights.matMul(V); // [1024, 16]
    const projected = attnOut.matMul(this.Wo).add(this.bo); // [1024, 16]

    // Residual connection
    x = x.add(projected) as tf.Tensor2D;

    // FFN with pre-norm
    const normed2 = this.layerNorm(x, this.ln2Gamma, this.ln2Beta);
    const ffnHidden = normed2.matMul(this.W1).add(this.b1).relu();
    const ffnOut = ffnHidden.matMul(this.W2).add(this.b2);

    // Residual connection
    x = x.add(ffnOut) as tf.Tensor2D;

    // Output head: [1024, 16] -> [1024, 1] -> [1024]
    const logits = x.matMul(this.Wout).add(this.bout).squeeze([1]);
    return logits as tf.Tensor1D;
  }

  /**
   * Train the model on one observed transition
   */
  private trainStep(gridState: number[], targetIndex: number): void {
    const target = tf.tensor1d([targetIndex], 'int32');

    this.optimizer.minimize(() => {
      const logits = this.forward(gridState);
      const logitsReshaped = logits.reshape([1, NUM_TOKENS]);
      const loss = tf.losses.softmaxCrossEntropy(
        tf.oneHot(target, NUM_TOKENS),
        logitsReshaped
      );
      return loss as tf.Scalar;
    }, true, this.allVariables);

    target.dispose();
  }

  /**
   * Build the flat grid state array from current environment state
   */
  private buildGridState(): number[] {
    const state = new Array(NUM_TOKENS).fill(0);
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
      // Train on the observed transition
      const targetIndex =
        prey.position[0] * GRID_SIZE + prey.position[1];
      this.trainStep(this.lastGridState, targetIndex);
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
