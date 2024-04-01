# Import the game

import Contra
from Contra.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

# Import the Joypad wrapper
from nes_py.wrappers import JoypadSpace
import numpy as np
import gym

JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

# Import Frame Stacker Wrapper and GrayScaling Wrapper
from gym.wrappers import GrayScaleObservation
# Import Vectorization Wrappers
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

# 1. Create the base environment
env = Contra.make('Contra-v0', apply_api_compatibility=True, render_mode = "human")
# 2. Simplify the controls 
env = JoypadSpace(env, SIMPLE_MOVEMENT+COMPLEX_MOVEMENT+RIGHT_ONLY)

# 3. Grayscale
env = GrayScaleObservation(env, keep_dim=True)
# 4. Wrap inside the Dummy Environment
env = DummyVecEnv([lambda: env])
# 5. Stack the frames
env = VecFrameStack(env, 4, channels_order='last')

state = env.reset()

state, reward, done, info = env.step([5])

# Import os for file path management
import os 
# Import PPO for algos
from stable_baselines3 import PPO
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True
    
CHECKPOINT_DIR = './Contra_RL/train'
LOG_DIR = './Contra_RL/logs/'

# Setup model saving callback
callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)

# This is the AI model started
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.000001, 
            n_steps=1024, batch_size = 32) 

# Load model
# You can use the most recent model by changing directory to './train/thisisthemostcurrenttmodel'
model.set_parameters('./Contra_RL/train/best_model_400000')

# Start the game 
state = env.reset()

# Loop through the game
while True: 
    action, _ = model.predict(state) 
    state, reward, done, info = env.step(action)
    env.render()