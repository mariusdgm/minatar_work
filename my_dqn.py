# save maximum Q value in training , loss

# q: should add environment seed? What about torch-deterministic -> yes, enable this, log it but don't always use the same one
# q: do you use wandb? -> maybe, not now
# q: epoch? -> use this
# TODO: split progress saving into logging, checkpoint, replay buffer (don't save replay buffer like this, split)


import time
import datetime
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s -%(message)s",
    datefmt="%H:%M:%S",
)


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
        self, env=None, output_file=None, checkpoints=True, load_file_path=None
    ) -> None:

        self.output_file_name = output_file

        self.store_intermediate_result = checkpoints
        self.checkpoint_file_name = load_file_path
        self.train_step_cnt = 200_000
        self.checkpoint_frequency = 50_000

        self.env = env
        self.epsilon_by_frame = self._get_linear_decay_function(
            start=1.0, end=0.01, decay=250_000
        )

        self.gamma = 0.99  # discount rate
        self.replay_buffer = ReplayBuffer(100_000)
        self.replay_start_size = 5000
        self.batch_size = 32
        self.training_freq = 4
        self.target_model_update_freq = 100

        self.validation_enabled = True
        self.validation_step_cnt = 100_000

        self.validation_epslion = 0.001

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
        self.t = 0  # frame nr
        self.e = 0  # episode nr
        self.policy_model_update_counter = 0
        self.avg_episode_rewards = []
        self.avg_episode_nr_frames = []  # how many frames did the episode last
        self.log_frame_stamp = []

        self.val_avg_episode_rewards = []
        self.val_avg_episode_nr_frames = []
        self.val_log_frame_stamp = []

        self.reset_episode_training_logs()

        if self.checkpoint_file_name is not None and os.path.exists(
            self.checkpoint_file_name
        ):
            self.load_training_state(self.checkpoint_file_name)

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

    def reset_episode_training_logs(self):
        self.episode_rewards = []
        self.episode_nr_frames = []

    def load_training_state(self, checkpoint_load_path):
        checkpoint = torch.load(checkpoint_load_path)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.policy_model.train()
        self.target_model.train()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.replay_buffer = checkpoint["replay_buffer"]
        self.t = checkpoint["frame"]
        self.e = checkpoint["episode"]
        self.policy_model_update_counter = checkpoint["policy_model_update_counter"]
        self.avg_episode_rewards = checkpoint["avg_episode_rewards"]
        self.avg_episode_nr_frames = checkpoint["avg_episode_nr_frames"]
        self.log_frame_stamp = checkpoint["log_frame_stamp"]

        self.val_avg_episode_rewards = checkpoint["val_avg_episode_rewards"]
        self.val_avg_episode_nr_frames = checkpoint["val_avg_episode_nr_frames"]
        self.val_log_frame_stamp = checkpoint["val_avg_episode_nr_frames"]

    def load_policy_model(self, model_path):
        model_data = torch.load(model_path)
        self.policy_model.load_state_dict(model_data["policy_model_state_dict"])

    def select_action(self, state, t, num_actions, epsilon=None, random_action=False):
        # A uniform random policy
        if random_action:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
            return action

        # Epsilon-greedy behavior policy for action selection
        if not epsilon:
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

    def train(self, train_frames, episode_termination_limit):
        logging.info(f"Starting/resuming training at: {self.t}")

        # Train for a number of episodes
        while self.t < train_frames:

            current_episode_reward, ep_frames = self.train_episode(
                train_frames, episode_termination_limit
            )

            self.episode_rewards.append(current_episode_reward)
            self.episode_nr_frames.append(ep_frames)
            self.e += 1

            if (
                self.validation_enabled
                and (self.t % self.train_step_cnt == 0)
                and (
                    self.t
                    > (
                        0
                        if not self.val_log_frame_stamp
                        else self.val_log_frame_stamp[-1]
                    )
                )  # check if this validation was already done
            ):
                logging.info(f"Starting validation at t = {self.t}")

                # start a validation sequence
                max_ep_reward, avg_ep_reward, avg_ep_nr_frames = self.validate_model(
                    episode_termination_limit
                )
                self.log_validation_info(max_ep_reward, avg_ep_reward, avg_ep_nr_frames)

                if self.store_intermediate_result:
                    self.save_training_status(self.checkpoint_file_name)

        # Print final logging info
        self.log_training_info()

        # Write data to file
        self.save_training_status(
            self.output_file_name,
        )

    def validate_model(self, episode_termination_limit):
        local_val_avg_episode_rewards = []
        local_val_avg_episode_nr_frames = []
        valiation_t = 0
        while valiation_t < self.validation_step_cnt:
            current_episode_reward = 0.0
            ep_frames = 0

            # Initialize the environment and start state
            self.env.reset()
            s = get_state(self.env.state())

            is_terminated = False
            while (
                (not is_terminated)
                and ep_frames < episode_termination_limit
                and valiation_t < self.validation_step_cnt
            ):
                action = self.select_action(
                    s, self.t, self.num_actions, epsilon=self.validation_epslion
                )
                reward, is_terminated = self.env.act(action)
                reward = torch.tensor([[reward]], device=device).float()
                is_terminated = torch.tensor([[is_terminated]], device=device)
                s_prime = get_state(self.env.state())

                current_episode_reward += reward.item()
                ep_frames += 1

                # Continue the process
                s = s_prime
                valiation_t += 1

            local_val_avg_episode_rewards.append(current_episode_reward)
            local_val_avg_episode_nr_frames.append(ep_frames)

        avg_ep_reward = sum(local_val_avg_episode_rewards) / len(
            local_val_avg_episode_rewards
        )

        avg_ep_nr_frames = sum(local_val_avg_episode_nr_frames) / len(
            local_val_avg_episode_nr_frames
        )

        self.val_avg_episode_rewards.append(avg_ep_reward)
        self.val_avg_episode_nr_frames.append(avg_ep_nr_frames)
        self.val_log_frame_stamp.append(self.t)

        return max(local_val_avg_episode_rewards), avg_ep_reward, avg_ep_nr_frames

    def train_episode(self, train_frames, episode_termination_limit):
        current_episode_reward = 0.0
        ep_frames = 0

        # Initialize the environment and start state
        self.env.reset()
        s = get_state(self.env.state())

        is_terminated = False
        while (
            (not is_terminated)
            and ep_frames < episode_termination_limit
            and self.t < train_frames
        ):

            # Logging training progress
            if (self.t % self.checkpoint_frequency == 0) and (
                self.t > (0 if not self.log_frame_stamp else self.log_frame_stamp[-1])
            ):  # check that we did not already log this status
                # must check for edge case where training is restarted
               
                self.log_training_info()
                if self.store_intermediate_result:
                    self.save_training_status(self.checkpoint_file_name)

            # Check if training routine should be stopped for a validation routine
            if (
                self.validation_enabled
                and (self.t % self.train_step_cnt == 0)
                and (
                    self.t
                    > (0 if not self.val_log_frame_stamp else self.val_log_frame_stamp[-1])
                )
            ):
                logging.info(f"Ending train episode for validation at t = {self.t}")
                break

            action = self.select_action(s, self.t, self.num_actions)
            reward, is_terminated = self.env.act(action)
            reward = torch.tensor([[reward]], device=device).float()
            is_terminated = torch.tensor([[is_terminated]], device=device)
            s_prime = get_state(self.env.state())

            self.replay_buffer.add(s, action, reward, s_prime, is_terminated)

            # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
            if (
                self.t > self.replay_start_size
                and len(self.replay_buffer) >= self.batch_size
            ):

                # Train every n number of frames
                if self.t % self.training_freq == 0:
                    sample = self.replay_buffer.sample(self.batch_size)
                    self.policy_model_update_counter += 1
                    self.learn(sample)

                # Update the target network only after some number of policy network updates
                if (
                    self.policy_model_update_counter > 0
                    and self.policy_model_update_counter % self.target_model_update_freq
                    == 0
                ):
                    self.target_model.load_state_dict(self.policy_model.state_dict())

            current_episode_reward += reward.item()

            self.t += 1
            ep_frames += 1

            # Continue the process
            s = s_prime

        # end of episode, return episode statistics:
        return current_episode_reward, ep_frames

    def log_validation_info(self, max_reward, avg_reward, avg_ep_len):
        logging.info(
            "Validation"
            + " | Max reward: "
            + str(max_reward)
            + " | Avg reward: "
            + str(np.around(avg_reward, 2))
            + " | Avg frames (episode): "
            + str(avg_ep_len)
        )

    def log_training_info(self):
        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        max_reward = max(self.episode_rewards)
        avg_ep_len = sum(self.episode_nr_frames) / len(self.episode_nr_frames)

        self.reset_episode_training_logs()

        self.avg_episode_rewards.append(avg_reward)
        self.avg_episode_nr_frames.append(avg_ep_len)
        self.log_frame_stamp.append(self.t)

        logging.info(
            "Frames seen: "
            + str(self.t)
            + " | Episode: "
            + str(self.e)
            + " | Max reward: "
            + str(max_reward)
            + " | Avg reward: "
            + str(np.around(avg_reward, 2))
            + " | Avg frames (episode): "
            + str(avg_ep_len)
            + " | Epsilon: "
            + str(self.epsilon_by_frame(self.t))
        )

    def save_training_status(self, save_file_name):
        torch.save(
            {
                "policy_model_state_dict": self.policy_model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "replay_buffer": self.replay_buffer,
                "frame": self.t,
                "episode": self.e,
                "policy_model_update_counter": self.policy_model_update_counter,
                "avg_episode_rewards": self.avg_episode_rewards,
                "avg_episode_nr_frames": self.avg_episode_nr_frames,
                "log_frame_stamp": self.log_frame_stamp,
                "val_avg_episode_rewards": self.val_avg_episode_rewards,
                "val_avg_episode_nr_frames": self.val_avg_episode_nr_frames,
                "val_log_frame_stamp": self.val_log_frame_stamp,
            },
            save_file_name,
        )
        logging.info(f"Checkpoint saved at t = {self.t}")

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

        loss = F.mse_loss(q_value, expected_q_value)
        # loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


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
        action = agent.select_action(state, agent.t, agent.num_actions, epsilon=agent.validation_epslion)
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
    my_agent.train(train_frames=10_000_000, episode_termination_limit=100_000)


if __name__ == "__main__":
    # main()

    play_game_visual("breakout")
