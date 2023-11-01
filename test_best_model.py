import os
import time

from callback import TrainAndLoggingCallback
from game_env import WebGame
from stable_baselines3 import DQN

CHECKPOINT_DIR ='./train/'
LOG_DIR = './logs/'

callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)
# create env
env = WebGame()

# load model
model = DQN.load(os.path.join('train', 'best_model_1000'))

#  testing loop pretrained model
for episode in range(1):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(int(action))
        time.sleep(0.01)
        total_reward += reward

    print(f'Total reward for episode {episode} is {total_reward}')