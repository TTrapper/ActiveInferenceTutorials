import { PixiRenderer } from './visualization/renderer';
import { PredatorPreySimulation, LessonType } from './simulations/predator-prey';
import { StateTransitionSimulation } from './simulations/state_transition';
import { SimulationController } from './core/types';

/**
 * Available simulations/lessons
 */
enum SimulationType {
  STATE_TRANSITION = 'state_transition',
  PREDATOR_PREY_L2 = 'predator_prey_l2',
  PREDATOR_PREY_L3 = 'predator_prey_l3',
  PREDATOR_PREY_L4 = 'predator_prey_l4'
}

/**
 * Initialize the application
 */
function initialize() {
  // Get DOM elements
  const canvasContainer = document.getElementById('simulation-canvas');
  const startButton = document.getElementById('start-button') as HTMLButtonElement;
  const pauseButton = document.getElementById('pause-button') as HTMLButtonElement;
  const stepButton = document.getElementById('step-button') as HTMLButtonElement;
  const resetButton = document.getElementById('reset-button') as HTMLButtonElement;
  const lessonSelector = document.getElementById('lesson-selector') as HTMLSelectElement;
  const lessonDescription = document.getElementById('lesson-description');
  const visionRangeConfig = document.getElementById('vision-range-config');
  const visionRangeSlider = document.getElementById('vision-range-slider') as HTMLInputElement;
  const visionRangeValue = document.getElementById('vision-range-value');

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
    [SimulationType.STATE_TRANSITION]:
      `Lesson 1: Theoretical Foundations -
      Basic state transition model demonstrating how a generative model
      can predict future states based on transition probabilities.`,
    [SimulationType.PREDATOR_PREY_L2]:
      `Lesson 2: Predator-Prey Simulation -
      A predator uses active inference to locate and catch prey
      by updating its beliefs based on observations with a uniform generative model.`,
    [SimulationType.PREDATOR_PREY_L3]:
      `Lesson 3: Predator-Prey with Advanced Belief Updating -
      The predator now uses a Bayesian approach to update its belief based on
      observations and its learned model of the prey's movement patterns.`,
    [SimulationType.PREDATOR_PREY_L4]:
      `Lesson 4: Predator-Prey with World Model -
      The predator uses a Bayesian world model that learns all possible gridworld
      states, allowing it to model changes in prey movement based on position
      of walls and food.`
  };

  /**
   * Show or hide the vision range slider based on the simulation type
   */
  function toggleVisionRangeSlider(simulationType: SimulationType) {
    if (visionRangeConfig && visionRangeConfig instanceof HTMLElement) {
      if (simulationType === SimulationType.PREDATOR_PREY_L2 ||
          simulationType === SimulationType.PREDATOR_PREY_L3 ||
          simulationType === SimulationType.PREDATOR_PREY_L4) {
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
      case SimulationType.STATE_TRANSITION:
        currentSimulation = new StateTransitionSimulation();
        break;
      case SimulationType.PREDATOR_PREY_L2:
        currentSimulation = new PredatorPreySimulation(LessonType.LESSON_2);
        break;
      case SimulationType.PREDATOR_PREY_L3:
        currentSimulation = new PredatorPreySimulation(LessonType.LESSON_3);
        break;
      case SimulationType.PREDATOR_PREY_L4:
        currentSimulation = new PredatorPreySimulation(LessonType.LESSON_4);
        break;
    }

    // Update description
    if (lessonDescription) {
      lessonDescription.textContent = lessonDescriptions[simulationType];
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
      
      // Update predator vision range if we have a predator-prey simulation
      if (currentSimulation && 'setVisionRange' in currentSimulation) {
        (currentSimulation as any).setVisionRange(range);
      }
    });
  }

  // Set up event listeners for simulation controls
  startButton.addEventListener('click', () => {
    if (currentSimulation) {
      currentSimulation.start(300); // Update every 300ms
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
    SimulationType.STATE_TRANSITION;

  switchSimulation(initialLesson);

  console.log('Simulation initialized successfully!');
}

// Initialize when the DOM is ready
document.addEventListener('DOMContentLoaded', initialize);
