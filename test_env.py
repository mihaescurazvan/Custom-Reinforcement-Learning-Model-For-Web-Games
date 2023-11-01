import pytesseract
import pydirectinput
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete
from mss import mss
from game_env import WebGame

env = WebGame()
obs = env.get_observation()
plt.imshow(obs[0])
done, done_cap = env.get_done()
plt.imshow(done_cap)
plt.show()
print(done)

#  testing loop for 10 games
for episode in range(10):
    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        total_reward += reward

    print(f'Total reward for episode {episode} is {total_reward}')