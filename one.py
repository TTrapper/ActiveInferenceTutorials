"""
Learning Coach
Great! I'm glad to hear you're a hands-on learner. Coding solutions from scratch is a fantastic way to solidify your understanding of active inference. I'll make sure to include coding challenges and exercises throughout our learning journey.

Let's begin with the first subtopic: Theoretical Foundations.

Active inference is built upon the idea that the brain is constantly trying to minimize its "free energy."  Think of free energy as a measure of how surprised the brain is by its sensory experiences. The brain does this by maintaining a generative model of the world, which is essentially a set of beliefs about how the world works. It then tries to predict what it will sense next, and if its predictions don't match reality, it updates its generative model to better align with the world.

A key aspect of active inference is that it doesn't just passively receive sensory information; it actively seeks out information that will minimize its free energy. This is where the "active" part comes in.  For example, when you're looking for your keys, you don't just stand there waiting for them to magically appear. You actively move your head and eyes, exploring different locations until you find them. Your brain is using its generative model to predict where the keys might be, and then it's guiding your actions to test those predictions.

Here's a simple coding challenge to get you started:

Problem: Implement a basic generative model.

Details:

1. Choose a simple environment, such as a 1D grid world.
2. Define a set of states for your environment (e.g., locations on the grid).
3. Create a probabilistic model that describes the transitions between these states. This model represents your generative model.
4. Write a function that takes the current state as input and samples a next state based on your transition probabilities. This function simulates how your generative model predicts the next state of the world.
This exercise will help you understand how generative models work and how they can be used to make predictions about the world.  Don't worry about making it perfect, the goal is to start coding and experimenting with these ideas.
"""

import numpy as np

class GenerativeModel:
    def __init__(self, world_size):
        """
        Initializes the generative model with a given world size.

        Args:
            world_size: The number of states in the world.
        """
        self.world_size = world_size
        self.states = np.eye(world_size)  # One-hot representation of states
        self.transition_matrix = self._create_transition_matrix()

    def _create_transition_matrix(self):
        """
        Creates a random transition matrix.

        Returns:
            A normalized transition matrix.
        """
        matrix = np.random.uniform(size=(self.world_size, self.world_size))
        return matrix / np.sum(matrix, axis=1, keepdims=True)

    def sample_next_state(self, current_state):
        """
        Samples the next state given the current state.

        Args:
            current_state: A one-hot vector representing the current state.

        Returns:
            A one-hot vector representing the sampled next state.
        """
        current_state_id = np.argmax(current_state)
        probabilities = self.transition_matrix[current_state_id]
        next_state_id = np.random.choice(len(probabilities), p=probabilities)
        next_state = np.zeros_like(current_state)
        next_state[next_state_id] = 1
        return next_state

    def run_simulation(self, initial_state, num_steps):
        """
        Runs a simulation for a given number of steps.

        Args:
            initial_state: A one-hot vector representing the initial state.
            num_steps: The number of simulation steps.

        Returns:
            A list of states representing the trajectory of the simulation.
        """
        current_state = initial_state
        trajectory = [current_state]
        for _ in range(num_steps):
            current_state = self.sample_next_state(current_state)
            trajectory.append(current_state)
        return trajectory


# Example usage:
world_size = 6
model = GenerativeModel(world_size)
initial_state = np.zeros(world_size)
initial_state[0] = 1

num_steps = 10
trajectory = model.run_simulation(initial_state, num_steps)

for state in trajectory:
    print(state)
