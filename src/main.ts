import { PixiRenderer } from './visualization/renderer';
import { PredatorPreySimulation, LessonType } from './simulations/predator-prey';
import { PreyOnlySimulation } from './simulations/prey-only';
import { SimulationController } from './core/types';

/**
 * Available simulations/lessons
 */
enum SimulationType {
  PREY_ONLY = 'prey_only',
  PREDATOR_PREY_L2 = 'predator_prey_l2',
  PREDATOR_PREY_L3 = 'predator_prey_l3'
}

/**
 * Initialize the application
 */
function initialize() {
  // Get DOM elements
  const canvasContainer = document.getElementById('simulation-canvas');
  const startButton =
    document.getElementById('start-button') as HTMLButtonElement;
  const pauseButton =
    document.getElementById('pause-button') as HTMLButtonElement;
  const stepButton =
    document.getElementById('step-button') as HTMLButtonElement;
  const resetButton =
    document.getElementById('reset-button') as HTMLButtonElement;
  const lessonSelector =
    document.getElementById('lesson-selector') as HTMLSelectElement;
  const lessonDescription =
    document.getElementById('lesson-description');
  const visionRangeConfig =
    document.getElementById('vision-range-config');
  const visionRangeSlider =
    document.getElementById('vision-range-slider') as HTMLInputElement;
  const visionRangeValue =
    document.getElementById('vision-range-value');

  if (!canvasContainer) {
    console.error('Canvas container not found!');
    return;
  }

  // Create renderer
  const renderer = new PixiRenderer(canvasContainer, 600, 600);

  // Track current simulation
  let currentSimulation: SimulationController;

  // Lesson descriptions
  const lessonDescriptions = {
    [SimulationType.PREY_ONLY]:
      `Lesson 1: State Transitions \u2014
      A prey agent moves on the grid with per-position transition
      policies. Watch how each grid cell produces a different movement
      distribution \u2014 this is the generative model the predator will
      need to learn.`,
    [SimulationType.PREDATOR_PREY_L2]:
      `Lesson 2: Learning Transitions \u2014
      A predator is introduced and learns the prey\u2019s per-position
      movement patterns using a Bayesian world model. The state key
      is the prey\u2019s position only (1,024 states on a 32\u00d732 grid).`,
    [SimulationType.PREDATOR_PREY_L3]:
      `Lesson 3: State Space Explosion \u2014
      The predator\u2019s own position is added to the state key, giving
      1,024\u00d71,024 = 1,048,576 possible states. Learning is visibly
      slower \u2014 this demonstrates the curse of dimensionality.`
  };

  /**
   * Show or hide the vision range slider based on the simulation type
   */
  function toggleVisionRangeSlider(simulationType: SimulationType) {
    if (visionRangeConfig && visionRangeConfig instanceof HTMLElement) {
      if (simulationType === SimulationType.PREDATOR_PREY_L2 ||
          simulationType === SimulationType.PREDATOR_PREY_L3) {
        visionRangeConfig.style.display = 'flex';
      } else {
        visionRangeConfig.style.display = 'none';
      }
    }
  }

  /**
   * Switch between different simulations/lessons
   */
  function switchSimulation(simulationType: SimulationType) {
    // Pause any running simulation
    if (currentSimulation) {
      currentSimulation.pause();
    }

    // Create new simulation based on type
    switch (simulationType) {
      case SimulationType.PREY_ONLY:
        currentSimulation = new PreyOnlySimulation();
        break;
      case SimulationType.PREDATOR_PREY_L2:
        currentSimulation =
          new PredatorPreySimulation(LessonType.LESSON_2);
        break;
      case SimulationType.PREDATOR_PREY_L3:
        currentSimulation =
          new PredatorPreySimulation(LessonType.LESSON_3);
        break;
    }

    // Update description
    if (lessonDescription) {
      lessonDescription.textContent =
        lessonDescriptions[simulationType];
    }

    // Connect simulation to renderer
    currentSimulation.addStateChangeListener((state) => {
      renderer.update(state);
    });

    // Toggle vision range slider visibility
    toggleVisionRangeSlider(simulationType);

    // Reset button states
    startButton.disabled = false;
    pauseButton.disabled = true;
    stepButton.disabled = false;
    resetButton.disabled = false;

    // Initialize the simulation
    currentSimulation.reset();
  }

  // Set up lesson selector
  if (lessonSelector) {
    lessonSelector.addEventListener('change', () => {
      const selectedLesson = lessonSelector.value as SimulationType;
      switchSimulation(selectedLesson);
    });
  }

  // Set up vision range slider
  if (visionRangeSlider && visionRangeValue) {
    visionRangeSlider.addEventListener('input', () => {
      const range = parseInt(visionRangeSlider.value);
      visionRangeValue.textContent = range.toString();

      if (currentSimulation && 'setVisionRange' in currentSimulation) {
        (currentSimulation as any).setVisionRange(range);
      }
    });
  }

  // Set up event listeners for simulation controls
  startButton.addEventListener('click', () => {
    if (currentSimulation) {
      currentSimulation.start(300);
      startButton.disabled = true;
      pauseButton.disabled = false;
      stepButton.disabled = true;
      resetButton.disabled = false;
    }
  });

  pauseButton.addEventListener('click', () => {
    if (currentSimulation) {
      currentSimulation.pause();
      startButton.disabled = false;
      pauseButton.disabled = true;
      stepButton.disabled = false;
    }
  });

  stepButton.addEventListener('click', () => {
    if (currentSimulation) {
      currentSimulation.step();
    }
  });

  resetButton.addEventListener('click', () => {
    if (currentSimulation) {
      currentSimulation.reset();
      startButton.disabled = false;
      pauseButton.disabled = true;
      stepButton.disabled = false;
    }
  });

  // Initialize with the first lesson
  const initialLesson = lessonSelector ?
    (lessonSelector.value as SimulationType) :
    SimulationType.PREY_ONLY;

  switchSimulation(initialLesson);

  console.log('Simulation initialized successfully!');
}

// Initialize when the DOM is ready
document.addEventListener('DOMContentLoaded', initialize);
