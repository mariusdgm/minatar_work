import os
import time

from minatar import Environment
from minatar.gui import GUI
import tkinter as Tk

import torch
import numpy as np
import random

from my_dqn import get_state, Conv_QNet


def get_action_from_model(model, state):
    with torch.no_grad():
        return model(state).max(1)[1].view(1, 1)


def get_action_in_state(model, state, num_actions):
    if np.random.binomial(1, 0.001) == 1:
        action = torch.tensor([[random.randrange(num_actions)]], device="cpu")
    else:
        action = get_action_from_model(model, state)

    return action


### Watch the agent in play
def play_game_visual(game):

    env = Environment(game)

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_folder = os.path.join(proj_dir, "checkpoints", game)
    file_name = os.path.join(default_save_folder, game + "_model")

    state_shape = env.state_shape()

    in_features = (state_shape[2], state_shape[0], state_shape[1])
    in_channels = in_features[0]
    num_actions = env.num_actions()

    model = Conv_QNet(in_features, in_channels, num_actions)
    checkpoint = torch.load(file_name)
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
        action = get_action_in_state(model, state, num_actions)
        reward, is_terminated = env.act(action)

        game_reward.set(game_reward.get() + reward)

        if is_terminated:
            is_terminate.set(True)

        gui.update(30, game_step_visual)

    gui.update(0, game_step_visual)
    gui.run()


if __name__ == "__main__":
    play_game_visual("breakout")
