import time
import datetime
import torch
import random
import numpy as np
import os
from pathlib import Path
import argparse

import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from minatar import Environment

from replay_buffer import ReplayBuffer
from utils import seed_everything, setup_logger

# NICE TO HAVE: gpu device at: model, wrapper of environment (in my case it would be get_state...),
# maybe: replay buffer (recommendation: keep on cpu, so that the env can run on gpu in parallel for multiple experiments)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# TODO: learn about Liftoff at a future date :)

# recommended experiment structure:

# date -> game_model_parameter -> folder_seed -> logs, replay buffer, checkpoints, models, config file

# TODO
# change training iteration stop, instead of making reinit of env when validation starts, 
# instead use 2 evs, one for training and one for valiation epoch
# 
# next training epoch starts where it left off 

# TODO 
# TODO: increase network by adding one extra conv

class Conv_QNet(nn.Module):
    def __init__(self, in_features, in_channels, num_actions, width_multiplicator=1):
        super().__init__()

        self.in_features = in_features
        self.in_channels = in_channels
        self.num_actions = num_actions

        self.conv_out_size = width_multiplicator * 16

        # conv layers
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, self.conv_out_size, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(self.conv_out_size, self.conv_out_size, kernel_size=3, stride=1),
            nn.ReLU()
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
        x = x.float()
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
        self,
        env,
        model_file=None,
        replay_buffer_file=None,
        train_stats_file=None,
        save_checkpoints=True,
        logger=None,
    ) -> None:

        self.env = env

        self.model_file = model_file
        self.replay_buffer_file = replay_buffer_file
        self.train_stats_file = train_stats_file
        self.save_checkpoints = save_checkpoints
        self.logger = logger

        self.train_step_cnt = 200_000
        self.validation_enabled = True
        self.validation_step_cnt = 100_000
        self.validation_epslion = 0.001
        self.episode_termination_limit = 10_000

        self.replay_start_size = 5000
        self.epsilon_by_frame = self._get_linear_decay_function(
            start=1.0, end=0.01, decay=250_000, eps_decay_start=self.replay_start_size
        )
        self.gamma = 0.99  # discount rate

        # returns state as [w, h, channels]
        state_shape = env.state_shape()

        # permute to get batch, channel, w, h shape
        # specific to minatar
        self.in_features = (state_shape[2], state_shape[0], state_shape[1])
        self.in_channels = self.in_features[0]
        self.num_actions = env.num_actions()

        self.replay_buffer = ReplayBuffer(
            max_size=100_000,
            state_dim=self.in_features,
            action_dim=1,
            n_step=0,
        )
        self.batch_size = 32
        self.training_freq = 4
        self.target_model_update_freq = 100

        self._init_models()  # init policy, target and optim

        # Set initial values related to training and monitoring
        self.t = 0  # frame nr
        self.episodes = 0  # episode nr
        self.policy_model_update_counter = 0

        self.training_stats = []
        self.validation_stats = []

        # check that all paths were provided and that the files can be found
        if (
            self.model_file is not None
            and os.path.exists(self.model_file)
            and self.replay_buffer_file is not None
            and os.path.exists(self.replay_buffer_file)
            and self.train_stats_file is not None
            and os.path.exists(self.train_stats_file)
        ):
            self.load_training_state(
                self.model_file, self.replay_buffer_file, self.train_stats_file
            )

    def _get_exp_decay_function(self, start, end, decay):
        return lambda x: end + (start - end) * np.exp(-1.0 * x / decay)

    def _get_linear_decay_function(self, start, end, decay, eps_decay_start):
        """Return a function that enables getting the value of epsilon at step x.

        Args:
            start (float): start value of the epsilon function (x=0)
            end (float): end value of the epsilon function (x=decay_in)
            decay_in (int): how many steps to reach the end value
        """
        return lambda x: max(
            end, min(start, start - (start - end) * ((x - eps_decay_start) / decay))
        )

    def _check_path(self, var_name, path):
        if path is None:
            raise ValueError("Provide a path")

    def _init_models(self):
        self.policy_model = Conv_QNet(
            self.in_features, self.in_channels, self.num_actions
        )
        self.target_model = Conv_QNet(
            self.in_features, self.in_channels, self.num_actions
        )

        self.optimizer = optim.Adam(
            self.policy_model.parameters(), lr=0.0000625, eps=0.00015
        )

    def load_training_state(
        self, models_load_file, replay_buffer_file, training_stats_file
    ):
        self.load_models(models_load_file)
        self.policy_model.train()
        self.target_model.train()
        self.load_training_stats(training_stats_file)
        self.replay_buffer.load(replay_buffer_file)

    def load_models(self, models_load_file):
        checkpoint = torch.load(models_load_file)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def load_training_stats(self, training_stats_file):
        checkpoint = torch.load(training_stats_file)

        self.t = checkpoint["frame"]
        self.episodes = checkpoint["episode"]
        self.policy_model_update_counter = checkpoint["policy_model_update_counter"]

        self.training_stats = checkpoint["training_stats"]
        self.validation_stats = checkpoint["validation_stats"]

    def save_checkpoint(self, model_file, replay_buffer_file, training_stats_file):
        self.logger.info(f"Saving checkpoint at t = {self.t} ...")
        self.save_model(model_file)
        self.save_training_status(training_stats_file)
        self.replay_buffer.save(replay_buffer_file)
        self.logger.info(f"Checkpoint saved at t = {self.t}")

    def save_model(self, model_file):
        torch.save(
            {
                "policy_model_state_dict": self.policy_model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_file,
        )
        self.logger.debug(f"Models saved at t = {self.t}")

    def save_training_status(self, training_status_file):
        torch.save(
            {
                "frame": self.t,
                "episode": self.episodes,
                "policy_model_update_counter": self.policy_model_update_counter,
                "training_stats": self.training_stats,
                "validation_stats": self.validation_stats,
            },
            training_status_file,
        )
        self.logger.debug(f"Training status saved at t = {self.t}")

    def select_action(self, state, t, num_actions, epsilon=None, random_action=False):
        max_q = np.nan 

        # A uniform random policy
        if random_action:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
            return action

        # Epsilon-greedy behavior policy for action selection
        if not epsilon:
            epsilon = self.epsilon_by_frame(t)

        if np.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            action, max_q = self.get_max_q_and_action(state)

        return action, max_q

    def get_max_q_val_for_state(self, state):
        with torch.no_grad():
            return self.policy_model(state).max(1)[0].item()

    def get_q_val_for_action(self, state, action):
        with torch.no_grad():
            return torch.index_select(
                self.policy_model(state), 1, action.squeeze(0)
            ).item()

    def get_action_from_model(self, state):
        with torch.no_grad():
            return self.policy_model(state).max(1)[1].view(1, 1)

    def get_max_q_and_action(self, state):
        with torch.no_grad():
            maxq_and_action = self.policy_model(state).max(1)
            q_val = maxq_and_action[0].item()
            action = maxq_and_action[1].view(1, 1)
            return action, q_val

    def train(self, train_epochs):
        self.logger.info(f"Starting/resuming training session at: {self.t}")

        # Train for a number of episodes
        for epoch in range(train_epochs):
            start_time = datetime.datetime.now()

            ep_train_stats = self.train_epoch()
            self.display_training_epoch_info(ep_train_stats)
            self.training_stats.append(ep_train_stats)

            if self.validation_enabled:
                ep_validation_stats = self.validate_epoch()
                self.display_validation_epoch_info(ep_validation_stats)
                self.validation_stats.append(ep_validation_stats)

            if self.save_checkpoints:
                self.save_checkpoint(
                    self.model_file, self.replay_buffer_file, self.train_stats_file
                )

            end_time = datetime.datetime.now()
            epoch_time = end_time - start_time

            self.logger.info(f"Epoch {epoch} completed in {epoch_time}")
            self.logger.info("\n")


        self.logger.info(f"Ended training session after {train_epochs} epochs at t = {self.t}")

    def train_epoch(self):
        self.logger.info(f"Starting training epoch at t = {self.t}")
        epoch_t = 0
        episode_rewards = []
        episode_nr_frames = []
        policy_trained_times = 0
        target_trained_times = 0
        ep_losses = []
        ep_max_qs = []

        start_time = datetime.datetime.now()
        while epoch_t < self.train_step_cnt:
            (
                current_episode_reward,
                ep_frames,
                ep_policy_trained_times,
                ep_target_trained_times,
                ep_losses,
                ep_max_qs,
            ) = self.train_episode(
                epoch_t, self.train_step_cnt, self.episode_termination_limit
            )

            epoch_t += ep_frames
            episode_rewards.append(current_episode_reward)
            episode_nr_frames.append(ep_frames)
            self.episodes += 1
            policy_trained_times += ep_policy_trained_times
            target_trained_times += ep_target_trained_times

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_training_epoch_stats(
            episode_rewards,
            episode_nr_frames,
            policy_trained_times,
            target_trained_times,
            ep_losses,
            ep_max_qs,
            epoch_time,
        )
        return epoch_stats

    def compute_training_epoch_stats(
        self,
        episode_rewards,
        episode_nr_frames,
        policy_trained_times,
        target_trained_times,
        ep_losses,
        ep_max_qs,
        epoch_time,
    ):
        stats = {}

        stats["frame_stamp"] = self.t

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_losses"] = self.get_vector_stats(ep_losses)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)

        stats["policy_trained_times"] = policy_trained_times
        stats["target_trained_times"] = target_trained_times
        stats["epoch_time"] = epoch_time

        return stats

    def get_vector_stats(self, vector):
        stats = {}

        if len(vector) > 0:
            stats["min"] = np.nanmin(vector)
            stats["max"] = np.nanmax(vector)
            stats["mean"] = np.nanmean(vector)
            stats["median"] = np.nanmedian(vector)
            stats["std"] = np.nanstd(vector)

        else:
            stats["min"] = None
            stats["max"] = None
            stats["mean"] = None
            stats["median"] = None
            stats["std"] = None

        return stats

    def validate_epoch(self):
        self.logger.info(f"Starting validation epoch at t = {self.t}")

        episode_rewards = []
        episode_nr_frames = []
        valiation_t = 0

        start_time = datetime.datetime.now()
        while valiation_t < self.validation_step_cnt:
            (
                valiation_t,
                current_episode_reward,
                ep_frames,
                ep_max_qs,
            ) = self.validate_episode(valiation_t, self.episode_termination_limit)

            valiation_t += ep_frames
            episode_rewards.append(current_episode_reward)
            episode_nr_frames.append(ep_frames)

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_validation_epoch_stats(
            episode_rewards,
            episode_nr_frames,
            ep_max_qs,
            epoch_time,
        )
        return epoch_stats

    def compute_validation_epoch_stats(
        self,
        episode_rewards,
        episode_nr_frames,
        ep_max_qs,
        epoch_time,
    ):
        stats = {}

        stats["frame_stamp"] = self.t

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)
        stats["epoch_time"] = epoch_time

        return stats

    def validate_episode(self, valiation_t, episode_termination_limit):

        current_episode_reward = 0.0
        ep_frames = 0
        max_qs = []

        # Initialize the environment and start state
        self.env.reset()
        s = get_state(self.env.state())

        is_terminated = False
        while (
            (not is_terminated)
            and ep_frames < episode_termination_limit
            and (valiation_t + ep_frames)
            < self.validation_step_cnt  # can early stop episode if the frame limit was reached
        ):

            action, max_q = self.select_action(
                s, self.t, self.num_actions, epsilon=self.validation_epslion
            )
            reward, is_terminated = self.env.act(action)
            reward = torch.tensor([[reward]], device=device).float()
            is_terminated = torch.tensor([[is_terminated]], device=device)
            s_prime = get_state(self.env.state())

            max_qs.append(max_q)

            current_episode_reward += reward.item()

            ep_frames += 1

            # Continue the process
            s = s_prime

        # end of episode, return episode statistics:
        new_valiation_t = valiation_t + ep_frames

        return (
            new_valiation_t,
            current_episode_reward,
            ep_frames,
            max_qs,
        )

    def train_episode(self, epoch_t, train_frames, episode_termination_limit):
        current_episode_reward = 0.0
        ep_frames = 0
        policy_trained_times = 0
        target_trained_times = 0
        losses = []
        max_qs = []

        # Initialize the environment and start state
        self.env.reset()
        s = get_state(self.env.state())

        is_terminated = False
        while (
            (not is_terminated)
            and ep_frames < episode_termination_limit
            and (epoch_t + ep_frames)
            < train_frames  # can early stop episode if the frame limit was reached
        ):

            action, max_q = self.select_action(s, self.t, self.num_actions)
            reward, is_terminated = self.env.act(action)
            reward = torch.tensor([[reward]], device=device).float()
            is_terminated = torch.tensor([[is_terminated]], device=device)
            s_prime = get_state(self.env.state())

            self.replay_buffer.append(s, action, reward, s_prime, is_terminated)
      
            max_qs.append(max_q)

            # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
            if (
                self.t > self.replay_start_size
                and len(self.replay_buffer) >= self.batch_size
            ):
                # Train every training_freq number of frames
                if self.t % self.training_freq == 0:
                    sample = self.replay_buffer.sample(self.batch_size)
                    self.policy_model_update_counter += 1
                    loss_val = self.model_learn(sample)

                    losses.append(loss_val)
                    policy_trained_times += 1

                # Update the target network only after some number of policy network updates
                if (
                    self.policy_model_update_counter > 0
                    and self.policy_model_update_counter % self.target_model_update_freq
                    == 0
                ):
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                    target_trained_times += 1

            current_episode_reward += reward.item()

            self.t += 1
            ep_frames += 1

            # Continue the process
            s = s_prime

        # end of episode, return episode statistics:
        return (
            current_episode_reward,
            ep_frames,
            policy_trained_times,
            target_trained_times,
            losses,
            max_qs,
        )

    def display_training_epoch_info(self, stats):
        self.logger.info(
            "TRAINING STATS" 
            + " | Frames seen: "
            + str(self.t)
            + " | Episode: "
            + str(self.episodes)
            + " | Max reward: "
            + str(stats["episode_rewards"]["max"])
            + " | Avg reward: "
            + str(stats["episode_rewards"]["mean"])
            + " | Avg frames (episode): "
            + str(stats["episode_frames"]["mean"])
            + " | Avg max Q: "
            + str(stats["episode_max_qs"]["mean"])
            + " | Epsilon: "
            + str(self.epsilon_by_frame(self.t))
            + " | Train epoch time: "
            + str(stats["epoch_time"])
        )

    def display_validation_epoch_info(self, stats):
        self.logger.info(
            "VALIDATION STATS" 
            + " | Max reward: "
            + str(stats["episode_rewards"]["max"])
            + " | Avg reward: "
            + str(stats["episode_rewards"]["mean"])
            + " | Avg frames (episode): "
            + str(stats["episode_frames"]["mean"])
            + " | Avg max Q: "
            + str(stats["episode_max_qs"]["mean"])
            + " | Validation epoch time: "
            + str(stats["epoch_time"])
        )

    def model_learn(self, sample):
        state, action, reward, next_state, terminated = sample

        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        terminated = torch.FloatTensor(terminated).unsqueeze(1)

        q_values = self.policy_model(state)
        selected_q_value = q_values.gather(1, action)

        next_q_values = self.target_model(next_state).detach()
        next_q_values = next_q_values.max(1)[0].unsqueeze(1)
        expected_q_value = reward + self.gamma * next_q_values * (1 - terminated)

        loss = F.mse_loss(selected_q_value, expected_q_value)
        # loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str, default="freeway")
    parser.add_argument("--checkpoint_folder", "-c", type=str)
    parser.add_argument("--save", "-s", action="store_true", default=True)
    args = parser.parse_args()

    proj_dir = os.path.dirname(os.path.abspath(__file__))
    default_checkpoint_folder = os.path.join(proj_dir, "checkpoints", args.game)

    if args.checkpoint_folder:
        checkpoint_folder = args.checkpoint_folder
    else:
        checkpoint_folder = default_checkpoint_folder

    model_file_name = os.path.join(checkpoint_folder, args.game + "_model")
    replay_buffer_file = os.path.join(checkpoint_folder, args.game + "_replay_buffer")
    train_stats_file = os.path.join(checkpoint_folder, args.game + "_train_stats")
    logs_path = os.path.join(checkpoint_folder, "logs")

    Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
    Path(logs_path).mkdir(parents=True, exist_ok=True)

    env = Environment(args.game)

    train_logger = setup_logger(args.game, logs_path)

    # print("Cuda available?: " + str(torch.cuda.is_available()))
    my_agent = AgentDQN(
        env=env,
        model_file=model_file_name,
        replay_buffer_file=replay_buffer_file,
        train_stats_file=train_stats_file,
        save_checkpoints=args.save,
        logger=train_logger,
    )
    my_agent.train(train_epochs=50)

    handlers = my_agent.logger.handlers[:]
    for handler in handlers:
        my_agent.logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    
    seed_everything(0)
    main()
    # play_game_visual("breakout")
