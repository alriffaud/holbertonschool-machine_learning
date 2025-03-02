#!/usr/bin/env python3
"""
Script to display a game played by a DQN agent using a trained policy for Atari
Breakout.
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
from PIL import Image
import gymnasium as gym
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, InputLayer, Reshape
from rl.memory import SequentialMemory
from rl.callbacks import Callback
try:
    import google.colab
    import time
    from IPython import display
    import imageio
    from IPython.display import display, clear_output, Image as IPyImage
    IN_COLAB = True
except ImportError:
    IN_COLAB = False


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
        return self.env.render()


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
        model (keras.models.Sequential): CNN model.
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


class VisualizationCallback(Callback):
    """
    Callback for visualizing the agent's gameplay using IPython.display.
    """
    def __init__(self, env, delay=0.02):
        """
        Initializes the VisualizationCallback.
        Args:
            env (gym.Env): The environment being visualized.
            delay (float): Time (in seconds) to pause between rendering frames.
        """
        self.env = env
        self.delay = delay
        self.episode_frames = []  # List to store frames for GIFs

    def on_action_end(self, action, logs={}):
        """
        Executes after an agent action and renders the frame using
        IPython.display.
        Args:
            action (int): The action performed by the agent.
            logs (dict): Training-related logs (optional).
        """
        # Capture the current frame in 'rgb_array' mode
        frame = self.env.render(mode='rgb_array')
        self.episode_frames.append(frame)

    def on_episode_end(self, episode, logs={}):
        """
        Executes after an episode ends, providing a short pause.
        Args:
            episode (int): The episode number that just finished.
            logs (dict): Training-related logs (optional).
        """
        # At the end of the episode, generate a GIF and show it
        gif_filename = f"episode_{episode}.gif"
        imageio.mimsave(gif_filename, self.episode_frames, fps=20)

        display(IPyImage(filename=gif_filename))

        # Reset the frame list for the next episode
        self.episode_frames = []
        time.sleep(1)  # Pause between episodes


def main():
    # Set render_mode based on environment
    render_mode = 'rgb_array' if IN_COLAB else 'human'
    # Create Atari Breakout environment with rendering enabled for human
    # visualization
    env = gym.make('ALE/Breakout-v5', render_mode=render_mode)
    # Apply custom preprocessing
    env = CustomAtariPreprocessing(env)
    # Wrap the environment to adapt its API for keras-rl
    env = GymnasiumWrapper(env)

    nb_actions = env.action_space.n
    # Since CustomAtariPreprocessing returns observations of shape (84,84,1)
    # and we use SequentialMemory with window_length=4, the state shape is
    # (4,84,84,1)
    input_shape = (4, 84, 84, 1)

    # Build the same model architecture as used in training
    model = build_model(input_shape, nb_actions)
    # Memory is required by the agent even during testing,
    # though it is not used for updates.
    memory = SequentialMemory(limit=500000, window_length=4)

    # Use a greedy policy for testing (always choose the best action)
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   policy=policy, nb_steps_warmup=0,
                   target_model_update=1, gamma=0.99)
    dqn.compile(optimizer='adam', metrics=['mae'])
    # Load the previously trained weights
    dqn.load_weights('policy.h5')

    if IN_COLAB:
        # Use the VisualizationCallback for rendering the game
        visualization_callback = VisualizationCallback(env, delay=0.02)
        scores = dqn.test(env, nb_episodes=5, visualize=False,
                          callbacks=[visualization_callback])
    else:
        # Test the agent for a fixed number of episodes (e.g., 5 episodes)
        dqn.test(env, nb_episodes=5, visualize=True)

    # Close the environment after testing
        env.close()


if __name__ == '__main__':
    main()
