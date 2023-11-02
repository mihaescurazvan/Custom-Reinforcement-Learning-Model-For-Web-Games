# Reinforcement Learning for Chrome Dino Game


Welcome to the Reinforcement Learning project for the Chrome Dino Game! In this project, we have developed a custom environment for training reinforcement learning agents to play the Chrome Dino Game, popularly known as the "T-Rex Runner" game.

## Project Overview

The project is structured as follows:

- `game_env.py`: Contains the `WebGame` class, which defines the game environment. This class implements essential methods for reinforcement learning, such as `step(self, action)`, `render(self)`, `reset(self, seed=0)`, `get_observation(self)`, `get_done(self)`, and `close(self)`. The `WebGame` class is responsible for interacting with the Chrome Dino Game, capturing screen images, performing game actions, and providing observations.

- `test_env.py`: A testing script to validate the correctness of the game environment. We use `env_checker` from Stable Baselines3 to ensure that the environment is set up correctly.

- `dqn.py`: This script is used to create and train a Deep Q-Network (DQN) model for the reinforcement learning task. We employ the DQN algorithm from Stable Baselines3 to train an agent to play the game.

- `test_best_model.py`: A script to load the best-trained model saved during training and test its performance. The model is evaluated to see how well it can play the Chrome Dino Game.

## Environment

The game environment implemented in `game_env.py` provides the following key components:

- **Action Space**: The action space is defined using Gym's `gym.spaces`. It allows the agent to take actions, such as jumping (space key), crouching (down key), or taking no action.

- **Observation Space**: The observation space captures the game's state as a screenshot of the screen. It is used as input for the reinforcement learning agent.

- **Screen Capturing**: The project uses `mss` to capture screen images, which are then processed to extract game information and the current game state.

- **Action Execution**: Game actions, such as jumping or crouching, are performed using `pyDirectInput`. This allows the agent to interact with the game by simulating keyboard input.

- **Text Recognition**: `pytesseract` is employed for Optical Character Recognition (OCR) to identify the "Game Over" text. When the game is over, it triggers a reset.

## Training and Testing

The training process is initiated in the `dqn.py` script using the Stable Baselines3 DQN implementation. Currently, a model is trained for only 2000 steps. For optimal performance, further training, likely on the order of 100,000 steps or more, is recommended.

The `test_best_model.py` script loads the best-trained model to evaluate its performance. It tests how well the reinforcement learning agent can play the Chrome Dino Game.

## Getting Started

To get started with this project, follow these steps:

1. Clone this GitHub repository to your local machine.

2. Install the required dependencies. You can find the necessary packages in the `requirements.txt` file.

3. Run the `dqn.py` script to initiate training for the reinforcement learning agent.

4. After training, use the `test_best_model.py` script to evaluate the trained model's performance.

5. Customize the environment and training parameters in the scripts to further optimize your agent's performance.

## Acknowledgments

This project is made possible by the contributions of the open-source community and the following libraries:

- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [pyDirectInput](https://pypi.org/project/PyDirectInput/)
- [mss](https://github.com/BoboTiG/python-mss)
- [pytesseract](https://github.com/madmaze/pytesseract)

## Note

Please note that the current model training is limited to 2000 steps for demonstration purposes. To achieve better performance, you are encouraged to train the model for a more substantial number of steps.

Feel free to contribute to this project, provide feedback, or make improvements. Happy gaming and happy training!
