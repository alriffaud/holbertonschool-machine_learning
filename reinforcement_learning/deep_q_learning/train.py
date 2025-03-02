#!/usr/bin/env python3
"""
Training script for a DQN agent to play Atari Breakout using keras-rl2 and
gymnasium.
"""
import tensorflow as tf
import sys
import importlib
# Ensure TensorFlow's Keras has a __version__ attribute
try:
    keras_module = sys.modules["tensorflow.keras"]
    keras_module.__version__ = tf.__version__
except KeyError:
    keras_module = importlib.import_module("tensorflow.keras")
    sys.modules["tensorflow.keras"] = keras_module
    keras_module.__version__ = tf.__version__
import numpy as np
import gymnasium as gym
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, InputLayer, Reshape
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy


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
        observation, _ = self.env.reset(**kwargs)
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


class CustomAtariPreprocessing(gym.ObservationWrapper):
    """
    Custom preprocessing for Atari games to replace AtariPreprocessing.
    - Converts frames to grayscale using Pillow.
    - Resizes frames to 84x84 using Pillow (nearest neighbor).
    - Skips frames manually (frame skipping is usually set in the base
    environment).
    """
    def __init__(self, env):
        """
        Initialize the wrapper.
        Args:
            env (gym.Env): Environment to wrap.
        """
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0,
                                                high=255,
                                                shape=(84, 84, 1),
                                                dtype=np.uint8)

    def observation(self, frame):
        """
        Preprocess the observation.
        Args:
            frame (np.ndarray): Frame to preprocess.
        Returns:
            np.ndarray: Preprocessed frame.
        """
        # Convert to grayscale and resize using PIL for better performance
        gray_frame = np.dot(frame[..., :3],
                            [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        pil_image = Image.fromarray(gray_frame)
        resized_frame = pil_image.resize((84, 84), Image.NEAREST)
        return np.array(resized_frame)[..., np.newaxis]


def build_model(input_shape, nb_actions):
    """
    Build a convolutional neural network model for the DQN agent.
    Args:
        input_shape (tuple): Shape of the input (height, width, channels).
        nb_actions (int): Number of possible actions.
    Returns:
        model (keras.models.Sequential): Compiled CNN model.
    """
    model = Sequential()
    # Input shape expected: (4,84,84,1)
    model.add(InputLayer(input_shape=input_shape))
    # Reshape from (4,84,84,1) to (84,84,4)
    model.add(Reshape((84, 84, 4)))
    # First convolutional layer: 16 filters, 8x8 kernel, stride 4
    model.add(Conv2D(16, (8, 8), strides=4, activation='relu'))
    # Second convolutional layer: 32 filters, 4x4 kernel, stride 2
    model.add(Conv2D(32, (4, 4), strides=2, activation='relu'))
    # Flatten the output for the fully connected layers
    model.add(Flatten())
    # Fully connected layer with 256 units
    model.add(Dense(256, activation='relu'))
    # Output layer: one neuron per possible action (linear activation)
    model.add(Dense(nb_actions, activation='linear'))
    return model


def main():
    # Create Atari Breakout environment with Gymnasium
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    # Apply custom preprocessing
    env = CustomAtariPreprocessing(env)
    # Wrap the environment to adapt its API for keras-rl
    env = GymnasiumWrapper(env)

    nb_actions = env.action_space.n
    # Since CustomAtariPreprocessing returns observations of shape (84,84,1)
    # and we use SequentialMemory with window_length=4, the state shape is
    # (4,84,84,1)
    input_shape = (4, 84, 84, 1)

    # Build the CNN model
    model = build_model(input_shape, nb_actions)
    print(model.summary())

    # Configure memory for experience replay
    memory = SequentialMemory(limit=500000, window_length=4)
    # Use epsilon-greedy policy for exploration during training
    policy = EpsGreedyQPolicy(eps=0.1)

    # Create the DQN agent with target network update and warm-up steps
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   policy=policy, nb_steps_warmup=50000,
                   target_model_update=10000, gamma=0.99, train_interval=4)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])

    # Train the agent
    # Note: For a full training one might need millions of frames
    dqn.fit(env, nb_steps=3000000, visualize=False, verbose=2)

    # Save the final policy network weights
    dqn.save_weights('policy.h5', overwrite=True)


if __name__ == '__main__':
    main()
