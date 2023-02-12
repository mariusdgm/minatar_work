import time
import datetime
import random

import torch
import torch.nn.utils.prune as prune

import numpy as np

import logging
import os
from pathlib import Path
import argparse

from minatar import Environment
from my_dqn import Conv_QNet
from utils import seed_everything, setup_logger

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


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


class PruningExperiment:
    def __init__(self, logger, game, model_params_file_name, exp_out_file):
        self.logger = logger
        self.game = game
        self.model_params_file_name = model_params_file_name
        self.exp_out_file = exp_out_file

        self.validation_step_cnt = 100_000
        self.validation_epslion = 0.001
        self.episode_termination_limit = 10_000

    def initialize_experiment(self):

        # get env dimensions
        self.env = Environment(self.game)
        state_shape = self.env.state_shape()

        self.in_features = (state_shape[2], state_shape[0], state_shape[1])
        self.in_channels = self.in_features[0]
        self.num_actions = self.env.num_actions()

        self.model = Conv_QNet(self.in_features, self.in_channels, self.num_actions)

        self.load_model_params()

    def load_model_params(self):
        checkpoint = torch.load(self.model_params_file_name)
        self.model.load_state_dict(checkpoint["policy_model_state_dict"])

    def prune_model_globally(self, pruning_amount):
        for name, module in self.model.named_modules():
            # structured pruning in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(
                    module, name="weight", amount=pruning_amount, n=2, dim=1
                )

            # unstructured pruning in linear layers
            elif isinstance(module, torch.nn.Linear):
                prune.l1_unstructured(module, name="weight", amount=pruning_amount)

    def get_vector_stats(self, vector):
        stats = {}

        if len(vector) > 0:
            stats["min"] = np.min(vector)
            stats["max"] = np.max(vector)
            stats["mean"] = np.mean(vector)
            stats["median"] = np.median(vector)
            stats["std"] = np.std(vector)

        else:
            stats["min"] = None
            stats["max"] = None
            stats["mean"] = None
            stats["median"] = None
            stats["std"] = None

        return stats

    def compute_validation_epoch_stats(
        self,
        episode_rewards,
        episode_nr_frames,
        ep_max_qs,
        epoch_time,
    ):
        stats = {}

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)
        stats["epoch_time"] = epoch_time

        return stats

    def select_action(self, state, num_actions, epsilon=None, random_action=False):
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

    def get_max_q_and_action(self, state):
        with torch.no_grad():
            maxq_and_action = self.model(state).max(1)
            q_val = maxq_and_action[0].item()
            action = maxq_and_action[1].view(1, 1)
            return action, q_val

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
                s, self.num_actions, epsilon=self.validation_epslion
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

    def validate_epoch(self):
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

    def save_experiment_results(self, experiment_results):
        torch.save(
            {"pruning_validation_results": experiment_results},
            self.exp_out_file,
        )

    def multiple_experiments(self, pruning_values):
        experiment_results = {}
        for pv in pruning_values:
            self.logger.info(
                f"Starting pruning experiment for {self.game} with pruning factor {pv}"
            )
            stats = self.single_experiment(pv)
            experiment_results[pv] = stats

        self.save_experiment_results(experiment_results)
        self.logger.info(
                f"Ended experiment for {self.game} ."
            )

    def single_experiment(self, pruning_value):

        self.initialize_experiment()
        if pruning_value > 0:
            self.prune_model_globally(pruning_value)

        validation_stats = self.validate_epoch()

        return validation_stats


def main():
    game = "freeway"

    # build path to trained model params
    proj_dir = os.path.abspath(".")
    default_save_folder = os.path.join(proj_dir, "checkpoints", game)
    params_file_name = os.path.join(default_save_folder, game + "_model")

    exp_out_folder = os.path.join(default_save_folder, "pruning_exp")
    Path(exp_out_folder).mkdir(parents=True, exist_ok=True)
    exp_out_file = os.path.join(exp_out_folder, "pruning_results")

    seed_everything(0)
    logger = setup_logger(game)

    pruning_experiment = PruningExperiment(logger, game, params_file_name, exp_out_file)

    pruning_experiment.multiple_experiments([0.00, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])

    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
