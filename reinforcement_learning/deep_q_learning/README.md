# Deep Q-Learning for Atari Breakout
<p align="center">
  <img src="https://i.imgur.com/5DYVnVE.jpg">
</p>

---

This project implements a Deep Q-Network (DQN) agent to play Atari Breakout using [keras-rl2](https://github.com/keras-rl/keras-rl) and [Gymnasium](https://gymnasium.farama.org/). The agent is trained using raw pixel data from the Atari emulator and uses experience replay along with a target network (handled internally by keras-rl2) for stable learning.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Assignment Requirements](#assignment-requirements)
- [Dependencies and Installation](#dependencies-and-installation)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
  - [Training](#training)
  - [Testing/Playing the Game](#testingplaying-the-game)
    - [Local (Visual Studio Code)](#local-visual-studio-code)
    - [Google Colab](#google-colab)
- [Detailed Explanation of train.py](#detailed-explanation-of-trainpy)
- [Detailed Explanation of play.py](#detailed-explanation-of-playpy)
- [Additional Notes](#additional-notes)
- [Author](#author)

---

## Project Overview

This project trains a DQN agent that learns to play Atari Breakout by directly processing raw pixel images. The network is a convolutional neural network that outputs Q-values for all possible actions. The training script (`train.py`) uses keras-rl2’s `DQNAgent`, `SequentialMemory`, and `EpsGreedyQPolicy` to train the agent and save the resulting policy weights to `policy.h5`. The testing script (`play.py`) loads the trained weights and runs the agent using a greedy policy (via `GreedyQPolicy`) to display gameplay.

---

## Assignment Requirements

- **General Requirements:**
  - Allowed editors: `vi`, `vim`, `emacs`.
  - All files will be interpreted/compiled on Ubuntu 20.04 LTS using Python 3.9.
  - Files will be executed with the following versions:
    - numpy 1.25.2
    - gymnasium 0.29.1
    - keras 2.15.0
    - keras-rl2 1.0.4
  - Every file must end with a new line.
  - The first line of every file must be exactly:  
    `#!/usr/bin/env python3`
  - A `README.md` file at the root of the project is mandatory.
  - The code must follow pycodestyle (version 2.11.1) and include documentation for all modules, classes, and functions.
  - All files must be executable and use the minimum number of operations.

- **Keras-RL Installation:**
  ```bash
  pip install --user keras-rl2==1.0.4
  ```

- **Dependencies:**
  ```bash
  pip install --user gymnasium[atari]==0.29.1
  pip install --user tensorflow==2.15.0
  pip install --user keras==2.15.0
  pip install --user numpy==1.25.2
  pip install --user Pillow==10.3.0
  pip install --user h5py==3.11.0
  pip install autorom[accept-rom-license]
  ```

- **Task: Breakout**
  - Write a Python script `train.py` that utilizes keras, keras-rl2, and gymnasium to train an agent that can play Atari’s Breakout.
    - Your script should utilize keras-rl2’s `DQNAgent`, `SequentialMemory`, and `EpsGreedyQPolicy`.
    - Your script should save the final policy network as `policy.h5`.
  - Write a Python script `play.py` that can display a game played by the agent trained by `train.py`:
    - Your script should load the policy network saved in `policy.h5`.
    - Your agent should use the `GreedyQPolicy`.
    - **HINT:** To make Gymnasium compatible with keras-rl, update the functions `reset`, `step` and `render` using custom wrappers.

---

## Dependencies and Installation

Ensure you have the following dependencies installed:

```bash
pip install --user gymnasium[atari]==0.29.1
pip install --user tensorflow==2.15.0
pip install --user keras==2.15.0
pip install --user numpy==1.25.2
pip install --user Pillow==10.3.0
pip install --user h5py==3.11.0
pip install autorom[accept-rom-license]
pip install --user keras-rl2==1.0.4
```

> **Note:** The project is developed and tested on Ubuntu 20.04 LTS using Python 3.9.

---

## Project Structure

```
reinforcement_learning/
└── deep_q_learning/
    ├── train.py      # Training script for the DQN agent
    ├── play.py       # Testing/playing script for the trained agent
    └── README.md     # This file
```

---

## How to Run

### Training

To train the DQN agent, run the `train.py` script. This script creates the Atari Breakout environment, applies custom preprocessing, builds the convolutional network, and trains the agent using keras-rl2’s `DQNAgent`. It saves the final model weights in a file called `policy.h5`.

```bash
chmod +x train.py
./train.py
```

> **Note:** Training may require millions of frames to converge. In the provided code, `nb_steps` is set to 3,000,000 for demonstration purposes. Adjust this value as needed.

### Testing/Playing the Game

The `play.py` script loads the trained weights from `policy.h5` and uses a greedy policy (`GreedyQPolicy`) to run the agent in the environment.

#### Local (Visual Studio Code)

When running locally, the script uses `render_mode='human'` to open a window displaying the game in real time.

```bash
chmod +x play.py
./play.py
```

#### Google Colab

Google Colab does not support opening new windows for rendering. Instead, the script is modified to accumulate rendered frames and display them as GIFs using IPython display. The script automatically detects if it is running in Colab and sets `render_mode` accordingly. Simply run the cell containing the `play.py` code in Colab, and the output will display GIFs for each episode along with textual information.

---

## Detailed Explanation of train.py

### Overview

The train.py script trains a DQN agent on Atari Breakout. It uses custom wrappers to preprocess images, adapts the Gymnasium API for compatibility with keras-rl2, builds a convolutional neural network, and configures the DQN agent with experience replay and an epsilon-greedy policy. The agent's weights are saved to policy.h5 once training is complete.

### Code Explanation

#### 1-Header and Setup

```python
#!/usr/bin/env python3
"""
Training script for a DQN agent to play Atari Breakout using keras-rl2 and gymnasium.
"""
import tensorflow as tf
import sys
import importlib
```

* Shebang and Docstring: Specifies the interpreter and describes the script.
* Imports: TensorFlow, sys, and importlib are used to ensure that tf.keras has the __version__ attribute.

#### 2-Keras Version Setup

```python
try:
    keras_module = sys.modules["tensorflow.keras"]
    keras_module.__version__ = tf.__version__
except KeyError:
    keras_module = importlib.import_module("tensorflow.keras")
    sys.modules["tensorflow.keras"] = keras_module
    keras_module.__version__ = tf.__version__
```

* Ensures that tf.keras has a version attribute, as required by keras-rl2.

#### Other Imports

```python
import numpy as np
import gymnasium as gym
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, InputLayer, Reshape
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy
```

* Importación de librerías esenciales: NumPy, Gymnasium, PIL, Keras y keras-rl2 components.

#### 4-GymnasiumWrapper

```python
class GymnasiumWrapper(gym.Wrapper):
    ...
    def reset(self, **kwargs):
        observation, _ = self.env.reset(**kwargs)
        return observation

    def step(self, action):
        (observation, reward, terminated, truncated, info) = self.env.step(action)
        done = terminated or truncated
        return observation, reward, done, info

    def render(self, mode='human'):
        self.env.render(mode)
```

* Adapta el entorno Gymnasium a la interfaz clásica de Gym (unificando los flags terminated y truncated).

#### 5-CustomAtariPreprocessing

```python
class CustomAtariPreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, frame):
        gray_frame = np.dot(frame[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
        pil_image = Image.fromarray(gray_frame)
        resized_frame = pil_image.resize((84, 84), Image.NEAREST)
        return np.array(resized_frame)[..., np.newaxis]
```

* Preprocesa cada frame: lo convierte a escala de grises, lo redimensiona a 84×84 y agrega una dimensión de canal.

#### 6-Model Building

```python
def build_model(input_shape, nb_actions):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Reshape((84, 84, 4)))
    model.add(Conv2D(16, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(32, (4, 4), strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model
```

* Construye una red convolucional:
  * InputLayer: Recibe un tensor con forma (4,84,84,1) (4 stacked frames).
  * Reshape: Transforma el tensor a (84,84,4) para adecuarse a la arquitectura.
  * Convolutions: Dos capas convolucionales extraen características visuales.
  * Flatten & Dense: Aplanamiento seguido de una capa densa y una capa de salida para predecir los Q-values.

#### 7-Main Function

```python
def main():
    env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')
    env = CustomAtariPreprocessing(env)
    env = GymnasiumWrapper(env)
    nb_actions = env.action_space.n
    input_shape = (4, 84, 84, 1)
    model = build_model(input_shape, nb_actions)
    print(model.summary())
    memory = SequentialMemory(limit=500000, window_length=4)
    policy = EpsGreedyQPolicy(eps=0.1)
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   policy=policy, nb_steps_warmup=50000,
                   target_model_update=10000, gamma=0.99, train_interval=4)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    dqn.fit(env, nb_steps=3000000, visualize=False, verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)
```

* Environment Creation: Crea el entorno Breakout con render_mode 'rgb_array'.
* Preprocessing & Wrapping: Se aplican los wrappers de preprocesamiento y adaptación.
* Parameters: Se define el número de acciones y la forma de entrada.
* Model & Memory: Se construye la red y se configura el buffer de memoria.
* Policy: Se utiliza EpsGreedyQPolicy con eps=0.1.
* DQN Agent: Se instancia y compila el agente DQN.
* Training: Se entrena el agente por 3,000,000 de pasos.
* Saving: Se guardan los pesos en policy.h5.

```python
if __name__ == '__main__':
    main()
```

* Garantiza que main() se ejecute cuando se corre el script.

---

## Detailed Explanation of play.py

### Overview

The play.py script is used to test and display the game played by the trained DQN agent. It loads the trained weights (policy.h5), rebuilds the same model architecture, and runs the agent in the environment using a greedy policy (via GreedyQPolicy). Additionally, it includes a visualization callback that is only activated in Google Colab, which accumulates frames and displays them as a GIF at the end of each episode.

### Code Explanation

#### 1-Header and Setup

```python
#!/usr/bin/env python3
"""
Script to display a game played by a DQN agent using a trained policy for Atari
Breakout.
"""
import tensorflow as tf
import sys
import importlib
```

* Similar to train.py, this sets up the interpreter and ensures that tf.keras has a __version__ attribute.

#### 2-Keras Version Setup

```python
try:
    keras_module = sys.modules["tensorflow.keras"]
    keras_module.__version__ = tf.__version__
except KeyError:
    keras_module = importlib.import_module("tensorflow.keras")
    sys.modules["tensorflow.keras"] = keras_module
    keras_module.__version__ = tf.__version__
```

* Ensures compatibility with keras-rl2.

#### 3-Additional Imports

```python
import numpy as np
from PIL import Image
import gymnasium as gym
from rl.agents.dqn import DQNAgent
from rl.policy import GreedyQPolicy
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, InputLayer, Reshape
from rl.memory import SequentialMemory
import time
from IPython import display
from rl.callbacks import Callback
import imageio
from IPython.display import display, clear_output, Image as IPyImage
```

* In addition to standard libraries, it imports modules for visualization (IPython.display, imageio) and a Callback from keras-rl2 to handle custom visualization in Colab.

#### 4-GymnasiumWrapper and CustomAtariPreprocessing

These classes are essentially the same as in train.py, adapting the environment and preprocessing frames.

#### 5-build_model

```python
def build_model(input_shape, nb_actions):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Reshape((84, 84, 4)))
    model.add(Conv2D(16, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(32, (4, 4), strides=2, activation='relu'))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model
```

* Rebuilds the CNN with the same architecture used during training.

#### 6-VisualizationCallback

```python
class VisualizationCallback(Callback):
    """
    Callback for visualizing the agent's gameplay using IPython.display.
    """
    def __init__(self, env, delay=0.02):
        self.env = env
        self.delay = delay
        self.episode_frames = []  # List to store frames

    def on_action_end(self, action, logs={}):
        frame = self.env.render(mode='rgb_array')
        self.episode_frames.append(frame)

    def on_episode_end(self, episode, logs={}):
        gif_filename = f"episode_{episode}.gif"
        imageio.mimsave(gif_filename, self.episode_frames, fps=20)
        display(IPyImage(filename=gif_filename))
        self.episode_frames = []
        time.sleep(1)
```

* on_action_end: Captures each frame rendered in 'rgb_array' mode and appends it to a list.
* on_episode_end: At the end of an episode, it saves the frames as a GIF, displays the GIF using IPython, and resets the frame list.

#### 7-Main Function

```python
def main():
    # Set render_mode based on environment
    render_mode = 'rgb_array' if IN_COLAB else 'human'
    env = gym.make('ALE/Breakout-v5', render_mode=render_mode)
    env = CustomAtariPreprocessing(env)
    env = GymnasiumWrapper(env)

    nb_actions = env.action_space.n
    input_shape = (4, 84, 84, 1)
    model = build_model(input_shape, nb_actions)
    memory = SequentialMemory(limit=500000, window_length=4)
    policy = GreedyQPolicy()

    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory,
                   policy=policy, nb_steps_warmup=0,
                   target_model_update=1, gamma=0.99)
    dqn.compile(optimizer='adam', metrics=['mae'])
    dqn.load_weights('policy.h5')

    if IN_COLAB:
        visualization_callback = VisualizationCallback(env, delay=0.02)
        scores = dqn.test(env, nb_episodes=5, visualize=False, callbacks=[visualization_callback])
    else:
        scores = dqn.test(env, nb_episodes=5, visualize=True)

    print('Average score over 5 test episodes:', np.mean(scores.history['episode_reward']))
    env.close()
```

* Dynamic Render Mode:
  The script checks if it is running in Google Colab (using IN_COLAB set by a try/except block). If so, it uses render_mode='rgb_array' and applies the VisualizationCallback   to display GIFs for each episode; otherwise, it uses render_mode='human' for a live display.
* Agent Setup:
  Rebuilds the model, memory, and loads the trained weights.
* Testing:
  Runs the agent for 5 episodes and prints the average score.

```python
if __name__ == '__main__':
    main()
```

* Ensures main() runs when the script is executed.

---

## Additional Notes

- **Training vs. Testing:**  
  The training script uses an epsilon-greedy policy (`EpsGreedyQPolicy`) with exploration, while the testing script uses a greedy policy (`GreedyQPolicy`) to evaluate the learned policy.
  
- **Rendering Differences:**  
  Local environments (e.g., Visual Studio Code) can open a window with `render_mode='human'`, while in Colab, a custom callback converts frames into a GIF for visualization.
  
- **Documentation & Style:**  
  Each module, class, and function is documented. The code adheres to pycodestyle guidelines and uses the minimum number of operations as required by the assignment.

---

## Author

- **Alberto Riffaud** - [GitHub](https://github.com/alriffaud) | [Linkedin](https://www.linkedin.com/in/alberto-riffaud) <br>

---

