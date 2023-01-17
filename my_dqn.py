import time
import torch
import random
import numpy as np
import copy
import logging
import os
from pathlib import Path
import argparse
from collections import deque, Counter, namedtuple

import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from minatar import Environment
from minatar.gui import GUI

import seaborn as sns
import matplotlib.pyplot as plt
import tkinter as Tk


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

logging.basicConfig(level=logging.INFO)


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        state = state.cpu().data
        next_state = next_state.cpu().data

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class Conv_QNet(nn.Module):
    def __init__(self, in_features, in_channels, num_actions):
        super().__init__()

        self.in_features = in_features
        self.in_channels = in_channels
        self.num_actions = num_actions

        # conv layers
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.size_linear_unit(), 128),
            nn.ReLU(),
            nn.Linear(128, self.num_actions),
        )

    def size_linear_unit(self):
        return (
            self.features(autograd.torch.zeros(*self.in_features)).view(1, -1).size(1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


def get_state(s):
    """
    Converts the state given by the environment to a tensor of size (in_channel, 10, 10), and then
    unsqueeze to expand along the 0th dimension so the function returns a tensor of size (1, in_channel, 10, 10).

    Args:
        s: current state as numpy array

    Returns:
        current state as tensor, permuted to match expected dimensions
    """
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


class AgentDQN:
    def __init__(
        self, env=None, output_file=None, intermediary_save=False, load_file_path=None
    ) -> None:

        self.output_file_name = output_file

        self.store_intermediate_result = intermediary_save
        self.checkpoint_file_name = load_file_path

        self.env = env
        self.epsilon_by_frame = self._get_linear_decay_function(
            start=1.0, end=0.01, decay=100000
        )

        self.gamma = 0.99  # discount rate
        self.replay_buffer = ReplayBuffer(100000)
        self.replay_start_size = 5000
        self.batch_size = 32
        self.training_freq = 4
        self.target_model_update_freq = 100

        # returns state as [w, h, channels]
        state_shape = env.state_shape()

        # permute to get batch, channel, w, h shape
        self.in_features = (state_shape[2], state_shape[0], state_shape[1])
        self.in_channels = self.in_features[0]
        self.num_actions = env.num_actions()

        self.policy_model = Conv_QNet(
            self.in_features, self.in_channels, self.num_actions
        )
        self.target_model = Conv_QNet(
            self.in_features, self.in_channels, self.num_actions
        )

        self.optimizer = optim.Adam(
            self.policy_model.parameters(), lr=0.0000625, eps=0.00015
        )

        # Set initial values related to training and monitoring
        self.e = 0  # episode nr
        self.t = 0  # frame nr
        self.policy_model_update_counter = 0

        self.data_return = []
        self.frame_stamp = []
        self.episode_nr_frames = []  # how many frames did the episode last

        if self.checkpoint_file_name is not None and os.path.exists(
            self.checkpoint_file_name
        ):
            self.load_training_state(self.checkpoint_file_name)

    def load_training_state(self, checkpoint_load_path):
        checkpoint = torch.load(checkpoint_load_path)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.policy_model.train()
        self.target_model.train()

        self.replay_buffer = checkpoint["replay_buffer"]

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.e = checkpoint["episode"]
        self.t = checkpoint["frame"]
        self.policy_model_update_counter = checkpoint["policy_model_update_counter"]
        self.episode_rewards = checkpoint["reward_per_run"]
        self.episode_nr_frames = checkpoint["episode_nr_frames"]
        self.frame_stamp = checkpoint["frame_stamp_per_run"]

    def load_policy_model(self, model_path):
        model_data = torch.load(model_path)
        self.policy_model.load_state_dict(model_data["policy_model_state_dict"])

    def select_action(self, state, t, num_actions):
        # A uniform random policy is run before the learning starts
        if t < self.replay_start_size:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)

        else:
            # Epsilon-greedy behavior policy for action selection
            epsilon = self.epsilon_by_frame(t - self.replay_start_size)
            if np.random.binomial(1, epsilon) == 1:
                action = torch.tensor([[random.randrange(num_actions)]], device=device)
            else:
                # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
                # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
                with torch.no_grad():
                    action = self.get_action_from_model(state)

        return action

    def get_action_from_model(self, state):
        return self.policy_model(state).max(1)[1].view(1, 1)

    def train(self, train_episodes, episode_termination_limit):

        t_start = time.time()

        max_reward = 0

        # Train for a number of episodes
        while self.e < train_episodes:
            # Initialize the return for every episode (we should see this eventually increase)
            current_episode_reward = 0.0
            ep_frames = 0

            # Initialize the environment and start state
            self.env.reset()
            s = get_state(self.env.state())

            is_terminated = False
            while (not is_terminated) and ep_frames < episode_termination_limit:
                # Generate data
                action = self.select_action(s, self.t, self.num_actions)

                reward, is_terminated = self.env.act(action)
                reward = torch.tensor([[reward]], device=device).float()
                is_terminated = torch.tensor([[is_terminated]], device=device)

                # Obtain next state
                s_prime = get_state(self.env.state())

                # Write the current frame to replay buffer
                self.replay_buffer.add(s, action, reward, s_prime, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if (
                    self.t > self.replay_start_size
                    and len(self.replay_buffer) >= self.batch_size
                ):
                    # Sample a batch
                    sample = self.replay_buffer.sample(self.batch_size)

                    # Train every n number of frames
                    if self.t % self.training_freq == 0:
                        self.policy_model_update_counter += 1
                        self.learn(sample)

                    # Update the target network only after some number of policy network updates
                    if (
                        self.policy_model_update_counter > 0
                        and self.policy_model_update_counter
                        % self.target_model_update_freq
                        == 0
                    ):
                        self.target_model.load_state_dict(
                            self.policy_model.state_dict()
                        )

                current_episode_reward += reward.item()
                if current_episode_reward > max_reward:
                    max_reward = current_episode_reward

                self.t += 1
                ep_frames += 1

                # Continue the process
                s = s_prime

            # Increment the episodes
            self.e += 1

            # Save the return for each episode
            self.episode_rewards.append(current_episode_reward)
            self.frame_stamp.append(self.t)
            self.episode_nr_frames.append(ep_frames)

            # Logging only when verbose is turned on and only at 1000 episode intervals
            if self.e % 1000 == 0:
                self.log_training_info(1000)

                if self.store_intermediate_result:
                    self.save_training_status(self.checkpoint_file_name)

        # Print final logging info
        self.log_training_info(1000)

        # Write data to file
        self.save_training_status(
            self.output_file_name,
        )

    def log_training_info(self, last_n):
        avg_reward = sum(self.episode_rewards[-last_n:]) / len(
            self.episode_rewards[-last_n:]
        )
        max_reward = max(self.episode_rewards[-last_n:])
        avg_episode_nr_frames = sum(self.episode_nr_frames[-last_n:]) / len(
            self.episode_nr_frames[-last_n:]
        )
        logging.info(
            "Episode "
            + str(self.e)
            + " | Max reward: "
            + str(max_reward)
            + " | Avg reward: "
            + str(np.around(avg_reward, 2))
            + " | Frames seen: "
            + str(self.t)
            + " | Avg frames (episode): "
            + str(avg_episode_nr_frames)
            + " | Time per frame: "
            + str((time.time() - t_start) / self.t)
        )

    def save_training_status(self, save_file_name):
        torch.save(
            {
                "episode": self.e,
                "frame": self.t,
                "policy_model_update_counter": self.policy_model_update_counter,
                "policy_model_state_dict": self.policy_model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "reward_per_run": self.episode_rewards,
                "frame_stamp_per_run": self.frame_stamp,
                "episode_nr_frames": self.episode_nr_frames,
                "replay_buffer": self.replay_buffer,
            },
            save_file_name,
        )

    def _get_exp_decay_function(self, start, end, decay):
        return lambda x: end + (start - end) * np.exp(-1.0 * x / decay)

    def _get_linear_decay_function(self, start, end, decay):
        """Return a function that enables getting the value of epsilon at step x.

        Args:
            start (float): start value of the epsilon function (x=0)
            end (float): end value of the epsilon function (x=decay_in)
            decay_in (int): how many steps to reach the end value
        """
        return lambda x: max(end, min(start, start - (start - end) * (x / decay)))

    def learn(self, sample):
        state, action, reward, next_state, terminated = sample

        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        terminated = torch.FloatTensor(terminated).unsqueeze(1)

        q_value = self.policy_model(state).gather(1, action)

        next_q_values = self.target_model(next_state).detach()
        next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        expected_q_value = reward + self.gamma * next_q_values * (1 - terminated)

        loss = (q_value - expected_q_value).pow(2).mean()
        # loss = F.smooth_l1_loss(expected_q_value, q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def play_game_visual(game):

    env = Environment(game)
    agent = AgentDQN(env=env)

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_folder = os.path.join(proj_dir, game)
    file_name = os.path.join(default_save_folder, game + "_model")

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
        action = agent.get_action_from_model(state)
        reward, is_terminated = env.act(action)

        game_reward.set(game_reward.get() + reward)

        if is_terminated:
            is_terminate.set(True)

        gui.update(50, game_step_visual)

    gui.update(0, game_step_visual)
    gui.run()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--save", "-s", action="store_true", default=True)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.game:
        game = args.game
    else:
        game = "breakout"

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_folder = os.path.join(proj_dir, game)

    if args.output:
        file_name = args.output
    else:
        file_name = os.path.join(default_save_folder, game + "_model")

    if args.loadfile:
        load_file_path = args.loadfile
    else:
        load_file_path = os.path.join(default_save_folder, game + "_checkpoint")

    if not args.output or not args.loadfile:
        Path(default_save_folder).mkdir(parents=True, exist_ok=True)

    env = Environment(game)

    # print("Cuda available?: " + str(torch.cuda.is_available()))
    my_agent = AgentDQN(env, file_name, args.save, load_file_path)
    my_agent.train(train_episodes=20000, episode_termination_limit=100000)


if __name__ == "__main__":
    # main()

    play_game_visual("breakout")
