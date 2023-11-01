import os
from callback import TrainAndLoggingCallback
from game_env import WebGame
from stable_baselines3 import DQN

CHECKPOINT_DIR ='./train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# create env
env = WebGame()
# create the DQN model
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1,
            buffer_size=1000000, learning_starts=1000)

model.learn(total_timesteps=5000, callback=callback)
