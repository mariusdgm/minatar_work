import os, sys
import time

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

from minatar import Environment
from minatar.gui import GUI
import tkinter as Tk

import torch
import numpy as np
import random

import yaml

from minatar_dqn.my_dqn import get_state, Conv_QNET


def get_action_from_model(model, state):
    with torch.no_grad():
        return model(state).max(1)[1].view(1, 1)


def get_action_in_state(model, state, num_actions, rand_chance=0.001):
    if np.random.binomial(1, rand_chance) == 1:
        action = torch.tensor([[random.randrange(num_actions)]], device="cpu")
    else:
        action = get_action_from_model(model, state)

    return action


### Watch the agent in play
def play_game_visual(model_path, config_path):

    with open(config_path, "r") as f:
        config_contents = yaml.safe_load(f)

    env = Environment(config_contents["environment"])

    state_shape = env.state_shape()

    in_features = (state_shape[2], state_shape[0], state_shape[1])
    in_channels = in_features[0]
    num_actions = env.num_actions()

    model = Conv_QNET(in_features, in_channels, num_actions, conv_hidden_out_size=config_contents["estimator"]["args_"]["conv_hidden_out_size"])
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["policy_model_state_dict"])

    gui = GUI(env.game_name(), env.n_channels)

    env.reset()

    is_terminate = Tk.BooleanVar()
    is_terminate.set(False)

    game_reward = Tk.DoubleVar()
    game_reward.set(0.0)

    def game_step_visual():

        if is_terminate.get() == True:
            print("Final Game score: ", str(game_reward.get()))
            time.sleep(3)
            game_reward.set(0.0)
            is_terminate.set(False)
            env.reset()

        gui.display_state(env.state())

        state = get_state(env.state())
        # state, t, num_actions, epsilon=None, random_action=False
        action = get_action_in_state(model, state, num_actions, rand_chance=0)
        reward, is_terminated = env.act(action)

        game_reward.set(game_reward.get() + reward)

        if is_terminated:
            is_terminate.set(True)

        gui.update(30, game_step_visual)

    gui.update(0, game_step_visual)
    gui.run()


if __name__ == "__main__":

    eperiment_path = r"D:\Work\PhD\minatar_work\experiments\training\outputs\2023_03_02-13_31_43\conv_model_32\breakout\0"
    model_file = r"conv_model_32_breakout_0_model"
    config_file = r"conv_model_32_breakout_0_config"

    model_path = os.path.join(eperiment_path, model_file)
    config_path = os.path.join(eperiment_path, config_file)

    play_game_visual(model_path, config_path)
