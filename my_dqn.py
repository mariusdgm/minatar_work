import time
import torch
import random
import numpy as np
import copy
import logging
import os
import argparse
from collections import deque, Counter, namedtuple

import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from minatar import Environment

import seaborn as sns
import matplotlib.pyplot as plt


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

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
            nn.Conv2d(self.in_channels, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
            nn.ReLU(),
            nn.Linear(256, self.num_actions),
        )

    def feature_size(self):
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
    def __init__(self, env, file_name, intermediary_save, load_file_path) -> None:

        self.output_file_name = file_name
        self.store_intermediate_result = False

        self.env = env
        self.epsilon_by_frame = self._get_linear_decay_function(
            start=1.0, end=0.01, decay=100000
        )

        self.gamma = 0.99  # discount rate
        self.replay_buffer = ReplayBuffer(100000)
        self.replay_start_size = 1000
        self.batch_size = 32
        self.training_freq = 4
        self.target_network_update_freq = 100

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

        # Set initial values related to training
        self.e_init = 0
        self.t_init = 0
        self.policy_net_update_counter_init = 0
        self.avg_return_init = 0.0
        self.data_return_init = []
        self.frame_stamp_init = []

        # TODO: implement loading mechanism

        self.t = self.t_init
        self.e = self.e_init

    def select_action(self, state, t, num_actions):
        # A uniform random policy is run before the learning starts
        # ASK: is this helpful?
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
                    action = self.policy_net(state).max(1)[1].view(1, 1)

        return action

    def train(self, train_episodes, episode_termination_limit):
        data_return = self.data_return_init
        frame_stamp = self.frame_stamp_init
        avg_return = self.avg_return_init

        # Train for a number of frames

        policy_net_update_counter = self.policy_net_update_counter_init
        t_start = time.time()
        while self.t < train_episodes:
            # Initialize the return for every episode (we should see this eventually increase)
            G = 0.0

            # Initialize the environment and start state
            self.env.reset()
            s = get_state(self.env.state())
            self.env.display_state(50)

            is_terminated = False
            while (not is_terminated) and self.t < episode_termination_limit:
                # Generate data
                action = self.select_action(s, self.t, self.num_actions)

                reward, terminated = self.env.act(action)
                reward = torch.tensor([[reward]], device=device).float()
                terminated = torch.tensor([[terminated]], device=device)

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
                        policy_net_update_counter += 1
                        self.learn(sample)

                    # Update the target network only after some number of policy network updates
                    if (
                        policy_net_update_counter > 0
                        and policy_net_update_counter % self.target_network_update_freq
                        == 0
                    ):
                        self.target_net.load_state_dict(self.policy_net.state_dict())

                G += reward.item()

                self.t += 1

                # Continue the process
                s = s_prime

            # Increment the episodes
            self.e += 1

            # Save the return for each episode
            data_return.append(G)
            frame_stamp.append(self.t)

            # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
            avg_return = 0.99 * avg_return + 0.01 * G
            if self.e % 1000 == 0:
                logging.info(
                    "Episode "
                    + str(self.e)
                    + " | Return: "
                    + str(G)
                    + " | Avg return: "
                    + str(np.around(avg_return, 2))
                    + " | Frame: "
                    + str(self.t)
                    + " | Time per frame: "
                    + str((time.time() - t_start) / self.t)
                )

            # Save model data and other intermediate data if the corresponding flag is true
            if self.store_intermediate_result and self.e % 1000 == 0:
                torch.save(
                    {
                        "episode": self.e,
                        "frame": self.t,
                        "policy_net_update_counter": policy_net_update_counter,
                        "policy_net_state_dict": self.policy_net.state_dict(),
                        "target_net_state_dict": self.target_net.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "avg_return": avg_return,
                        "return_per_run": data_return,
                        "frame_stamp_per_run": frame_stamp,
                        "replay_buffer": self.replay_buffer,
                    },
                    self.output_file_name + "_checkpoint",
                )

        # Print final logging info
        logging.info(
            "Avg return: "
            + str(np.around(avg_return, 2))
            + " | Time per frame: "
            + str((time.time() - t_start) / self.t)
        )

        # Write data to file
        torch.save(
            {
                "returns": data_return,
                "frame_stamps": frame_stamp,
                "policy_net_state_dict": self.policy_net.state_dict(),
            },
            self.output_file_name + "_data_and_weights",
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
        return lambda x: max(
            end, min(start, start - (start - end) * (x / decay))
        )

    def learn(self, sample):
        state, action, reward, next_state, terminated = sample

        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        terminated = torch.FloatTensor(terminated)

        q_value = self.policy_model(state).gather(1, action)

        next_q_values = self.target_model(next_state).detach()
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward + self.gamma * next_q_value * (1 - terminated)
        expected_q_value = torch.unsqueeze(expected_q_value, 1)

        loss = (q_value - expected_q_value).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--save", "-s", action="store_true", default=False)
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.game:
        game = args.game
    else:
        game = "breakout"

    # If there's an output specified, then use the user specified output. Otherwise, create file in the current
    # directory with the game's name.
    if args.output:
        file_name = args.output
    else:
        file_name = os.getcwd() + "/" + game

    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile

    env = Environment(game)

    print("Cuda available?: " + str(torch.cuda.is_available()))
    my_agent = AgentDQN(env, file_name, args.save, load_file_path)
    my_agent.train(train_episodes=10, episode_termination_limit=10000)


if __name__ == "__main__":
    main()
