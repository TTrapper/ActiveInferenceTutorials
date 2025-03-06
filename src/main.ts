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
  PREDATOR_PREY_L3 = 'predator_prey_l3'
}

/**
 * Initialize the application
 */
function initialize() {
  // Get DOM elements
  const canvasContainer = document.getElementById('simulation-canvas');
  const startButton = document.getElementById('start-button') as HTMLButtonElement;
  const pauseButton = document.getElementById('pause-button') as HTMLButtonElement;
  const resetButton = document.getElementById('reset-button') as HTMLButtonElement;
  const lessonSelector = document.getElementById('lesson-selector') as HTMLSelectElement;
  const lessonDescription = document.getElementById('lesson-description');
  const policyEditor = document.getElementById('policy-editor');
  const policyGrid = document.getElementById('policy-grid');
  const resetPolicyButton = document.getElementById('reset-policy-button') as HTMLButtonElement;

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
      observations and its learned model of the prey's movement patterns.`
  };

  /**
   * Initialize the policy editor
   */
  function initializePolicyEditor() {
    if (!policyGrid) return;

    // Clear existing grid
    policyGrid.innerHTML = '';

    // Create a 5x5 grid for the policy editor
    for (let x = 0; x < 5; x++) {
      for (let y = 0; y < 5; y++) {
        const cell = document.createElement('div');
        cell.className = 'policy-cell';
        cell.dataset.x = x.toString();
        cell.dataset.y = y.toString();

        // Center cell (2,2) represents the prey's current position
        if (x === 2 && y === 2) {
          cell.className += ' prey-cell';
          cell.innerHTML = '🐁'; // Prey icon
        } else {
          // Add click event to update policy
          cell.addEventListener('click', () => {
            if (currentSimulation && 'updatePreyPolicy' in currentSimulation) {
              (currentSimulation as any).updatePreyPolicy(x, y);
            }
          });

          // Add value span
          const valueSpan = document.createElement('span');
          valueSpan.className = 'policy-value';
          valueSpan.textContent = '0.00';
          cell.appendChild(valueSpan);
        }

        policyGrid.appendChild(cell);
      }
    }

    // Set up reset policy button
    if (resetPolicyButton) {
      resetPolicyButton.addEventListener('click', () => {
        if (currentSimulation && 'resetPreyPolicy' in currentSimulation) {
          (currentSimulation as any).resetPreyPolicy();
        }
      });
    }
  }

  /**
   * Update the policy editor with current policy values
   */
  function updatePolicyEditor(policy: number[][]) {
    if (!policyGrid) return;

    // Update each cell's value
    for (let y = 0; y < 5; y++) {
      for (let x = 0; x < 5; x++) {
        // Skip the center cell (prey position)
        if (x === 2 && y === 2) continue;

        const cell = policyGrid.querySelector(`[data-x="${x}"][data-y="${y}"]`);
        if (cell) {
          const valueSpan = cell.querySelector('.policy-value');
          if (valueSpan) {
            const probability = policy[y][x];
            valueSpan.textContent = probability.toFixed(2);

            // Update cell background color based on probability
            const intensity = Math.min(255, Math.round(probability * 500));
            cell.style.backgroundColor = `rgba(0, ${intensity}, 0, 0.2)`;
          }
        }
      }
    }
  }

  /**
   * Show or hide the policy editor based on the simulation type
   */
  function togglePolicyEditor(simulationType: SimulationType) {
    if (policyEditor && policyEditor instanceof HTMLElement) {
      if (simulationType === SimulationType.PREDATOR_PREY_L2 ||
          simulationType === SimulationType.PREDATOR_PREY_L3) {
        policyEditor.style.display = 'flex';
        initializePolicyEditor();
      } else {
        policyEditor.style.display = 'none';
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
    }

    // Update description
    if (lessonDescription) {
      lessonDescription.textContent = lessonDescriptions[simulationType];
    }

    // Connect simulation to renderer
    currentSimulation.addStateChangeListener((state) => {
      renderer.update(state);

      // Update policy editor if available
      if (state.preyMovementPolicy) {
        updatePolicyEditor(state.preyMovementPolicy);
      }
    });

    // Toggle policy editor visibility
    togglePolicyEditor(simulationType);

    // Reset button states
    startButton.disabled = false;
    pauseButton.disabled = true;
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

  // Set up event listeners for simulation controls
  startButton.addEventListener('click', () => {
    if (currentSimulation) {
      currentSimulation.start(300); // Update every 300ms
      startButton.disabled = true;
      pauseButton.disabled = false;
      resetButton.disabled = false;
    }
  });

  pauseButton.addEventListener('click', () => {
    if (currentSimulation) {
      currentSimulation.pause();
      startButton.disabled = false;
      pauseButton.disabled = true;
    }
  });

  resetButton.addEventListener('click', () => {
    if (currentSimulation) {
      currentSimulation.reset();
      startButton.disabled = false;
      pauseButton.disabled = true;
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
