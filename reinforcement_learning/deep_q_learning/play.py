#!/usr/bin/env python3
"""
Script to display a game played by a DQN agent using a trained policy for Atari
Breakout.
"""
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, InputLayer
from rl.memory import SequentialMemory


class GymnasiumWrapper(gym.Wrapper):
    """
    Wrapper to adapt a Gymnasium environment to the OpenAI Gym interface for
    keras-rl.
    """
    def reset(self, **kwargs):
        """
        Reset the environment and return the initial observation.
        Args:
            kwargs: Additional arguments to pass to the environment.
        Returns:
            observation (np.ndarray): Initial observation.
        """
        observation, info = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        """
        Execute the given action in the environment and return the result.
        Args:
            action (int): Action to execute.
        Returns:
            observation (np.ndarray): New observation.
            reward (float): Reward for the action.
            done (bool): Whether the episode is done.
            info (dict): Additional information.
        """
        (observation, reward, terminated,
         truncated, info) = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode='human'):
        """
        Render the environment.
        Args:
            mode (str): Rendering mode.
        """
        self.env.render(mode)


def build_model(input_shape, nb_actions):
    """
    Build a convolutional neural network model for the DQN agent.
    Args:
        input_shape (tuple): Shape of the input (height, width, channels).
        nb_actions (int): Number of possible actions.
    Returns:
        model (keras.models.Sequential): CNN model.
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    # First convolutional layer: 16 filters, 8x8 kernel, stride 4
    model.add(Conv2D(16, (8, 8), strides=4, activation='relu'))
    # Second convolutional layer: 32 filters, 4x4 kernel, stride 2
    model.add(Conv2D(32, (4, 4), strides=2, activation='relu'))
    model.add(Flatten())
    # Fully connected layer with 256 units
    model.add(Dense(256, activation='relu'))
    # Output layer: one neuron per action (linear activation)
    model.add(Dense(nb_actions, activation='linear'))
    return model


def main():
    # Create Atari Breakout environment with rendering enabled for human
    # visualization
    env = gym.make('ALE/Breakout-v5', render_mode='human')
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True,
                             scale_obs=False)
    env = FrameStack(env, num_stack=4)
    env = GymnasiumWrapper(env)

    nb_actions = env.action_space.n
    input_shape = env.observation_space.shape

    # Build the same model architecture as used in training
    model = build_model(input_shape, nb_actions)
    # Memory is required by the agent even during testing,
    # though it is not used for updates.
    memory = SequentialMemory(limit=1000000, window_length=4)

    # Use a greedy policy for testing (always choose the best action)
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   policy=policy, nb_steps_warmup=0,
                   target_model_update=1, gamma=0.99)
    dqn.compile(optimizer='adam', metrics=['mae'])
    # Load the previously trained weights
    dqn.load_weights('policy.h5')

    # Test the agent for a fixed number of episodes (e.g., 5 episodes)
    dqn.test(env, nb_episodes=5, visualize=True)


if __name__ == '__main__':
    main()
