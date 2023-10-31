import pytesseract
import pydirectinput
import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete
from mss import mss


# creating a custom environment
class WebGame(Env):
    def __init__(self):
        super().__init__()
        # setup spaces
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # define extraction parameters for the game
        self.cap = mss()
        self.game_location = {'top': 300, 'left': 0, 'width': 600, 'height': 500}
        self.done_location = {'top': 405, 'left': 630, 'width': 660, 'height': 70}

    # what is get called to do something ,in the game
    def step(self, action):
        # action key 0 = space, 1 = duck, 2 = no action
        action_space = {
            0: 'space',
            1: 'down',
            2: 'no_op'
        }
        if action != 2:
            pydirectinput.press(action[action_space[action]])

        # checking whether the game is done
        done, done_cap = self.get_done()
        # get the next observation
        new_observation = self.get_observation()
        # reward for every frame where alive
        reward = 1
        # info dictionary
        info = {}

        return new_observation, reward, done, info

    # visualize the game
    def render(self):
        cv2.imshow('Game', np.array(self.cap.grab(self.game_location))[:, :, :3])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    # restart the game
    def reset(self):
        time.sleep(1)
        pydirectinput.click(x=150, y=150)
        pydirectinput.press('space')
        return self.get_observation()


    # get the part of the observation of the game that we need
    def get_observation(self):
        # get scree catpure of game
        raw = np.array(self.cap.grab(self.game_location))[:, :, :3].astype(np.uint8)
        # greyscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray, (100, 83))
        # add chanels first
        channel = np.reshape(resized, (1, 83, 100))

        return channel

    # get the done text using OCR
    def get_done(self):
        # get done screen
        done_cap = np.array(self.cap.grab(self.done_location))[:, :, :3].astype(np.uint8)
        # valid done text
        done_strings = ['GAME', 'GAHE', 'GANE']

        done = False
        res = pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done = True

        return done, done_cap

    # this closes down the observation
    def close(self):
        cv2.destroyAllWindows()


env = WebGame()
print(env.action_space.sample())
print(plt.imshow(env.observation_space.sample()[0]))
print(env.get_observation().shape)