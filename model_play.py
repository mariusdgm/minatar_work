import os
import time

from minatar import Environment
from minatar.gui import GUI
import tkinter as Tk

from my_dqn import AgentDQN, get_state

### Watch the agent in play
def play_game_visual(game):

    env = Environment(game)
    agent = AgentDQN(env=env)

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_folder = os.path.join(proj_dir, game)
    file_name = os.path.join(default_save_folder, game + "_checkpoint")

    agent.load_policy_model(file_name)

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
        action = agent.select_action(
            state, agent.t, agent.num_actions, epsilon=agent.validation_epslion
        )
        reward, is_terminated = env.act(action)

        game_reward.set(game_reward.get() + reward)

        if is_terminated:
            is_terminate.set(True)

        gui.update(50, game_step_visual)

    gui.update(0, game_step_visual)
    gui.run()

if __name__ == "__main__":
    play_game_visual("breakout")
