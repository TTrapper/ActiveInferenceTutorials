import { Environment, SimulationController, SimulationState, StateChangeListener } from './types';

/**
 * Base simulation controller that manages environment and agents
 */
export class BaseSimulationController implements SimulationController {
  environment: Environment;
  running: boolean = false;
  intervalId: number | null = null;
  stateChangeListeners: StateChangeListener[] = [];

  constructor(environment: Environment) {
    this.environment = environment;
  }

  /**
   * Add listener for simulation state changes
   */
  addStateChangeListener(listener: StateChangeListener): void {
    this.stateChangeListeners.push(listener);
  }

  /**
   * Get current simulation state
   */
  getState(): SimulationState {
    // This is a base implementation
    // Specific simulations will override with their own state
    return {
      agents: [],
      environment: {
        type: 'base',
        size: 0
      }
    };
  }

  /**
   * Notify listeners of state change
   */
  notifyStateChange(): void {
    const state = this.getState();
    for (const listener of this.stateChangeListeners) {
      listener(state);
    }
  }

  /**
   * Perform one simulation step
   */
  step(): void {
    this.environment.step();
    this.notifyStateChange();
  }

  /**
   * Start simulation with given interval
   */
  start(intervalMs: number = 500): void {
    if (!this.running) {
      this.running = true;
      this.intervalId = window.setInterval(() => this.step(), intervalMs);
      
      // Ensure initial state is sent
      this.notifyStateChange();
    }
  }

  /**
   * Pause the simulation
   */
  pause(): void {
    if (this.running && this.intervalId !== null) {
      this.running = false;
      window.clearInterval(this.intervalId);
      this.intervalId = null;
    }
  }

  /**
   * Reset the simulation - to be implemented by specific simulations
   */
  reset(): void {
    this.pause();
    // Specific implementation will need to reset environment and agents
    this.notifyStateChange();
  }
}