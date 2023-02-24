import os
import sys

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

import datetime
import random
import multiprocessing


import torch
import torch.nn.utils.prune as prune

import numpy as np

from pathlib import Path
import argparse

from minatar import Environment
from minatar_dqn.my_dqn import Conv_QNET
from minatar_dqn.utils.my_logging import setup_logger

from experiments.experiment_utils import (
    seed_everything,
    search_files_with_string,
    split_path_at_substring,
)


os.environ["OMP_NUM_THREADS"] = "2"

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
    def __init__(
        self,
        logger,
        game,
        model_params_file_name,
        exp_out_file,
        pruning_function,
        experiment_info,
    ):
        self.logger = logger
        self.game = game
        self.model_params_file_name = model_params_file_name
        self.exp_out_file = exp_out_file
        self.pruning_function = pruning_function
        self.exp_info = experiment_info

        self.validation_step_cnt = 100_000
        self.validation_epsilon = 0.001
        self.episode_termination_limit = 10_000

    def initialize_experiment(self):

        seed_everything(0)

        # get env dimensions
        self.env = Environment(self.game, random_seed=0)
        state_shape = self.env.state_shape()

        self.in_features = (state_shape[2], state_shape[0], state_shape[1])
        self.in_channels = self.in_features[0]
        self.num_actions = self.env.num_actions()

        self.model = Conv_QNET(self.in_features, self.in_channels, self.num_actions)

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
                s, self.num_actions, epsilon=self.validation_epsilon
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
            {
                "pruning_validation_results": experiment_results,
                "experiment_info": self.exp_info,
            },
            self.exp_out_file,
        )

    def perform_multiple_experiments(self, pruning_values):
        experiment_results = {}
        for pv in pruning_values:
            self.logger.info(
                f"Starting pruning experiment for {self.game} with pruning method {self.pruning_function.__name__} using pruning factor {pv}"
            )
            stats = self.single_experiment(pv)
            experiment_results[pv] = stats

        self.save_experiment_results(experiment_results)
        self.logger.info(
            f"Ended experiment for {self.game} with pruning method {self.pruning_function.__name__}."
        )

    def perform_single_experiment(self, pruning_value):
        experiment_results = {}

        if self.pruning_function:
            self.logger.info(
                f"Starting pruning experiment for {self.game} with pruning method {self.pruning_function.__name__} using pruning factor {pruning_value}"
            )
        else:
            self.logger.info(
                f"No pruning function defined, running baseline experiment"
            )
        stats = self.single_experiment(pruning_value)
        experiment_results[pruning_value] = stats

        self.save_experiment_results(experiment_results)

        if self.pruning_function:
            self.logger.info(
                f"Ended experiment for {self.game} with pruning method {self.pruning_function.__name__}."
            )
        else:
            self.logger.info(f"Baseline experiment ended")

    def single_experiment(self, pruning_value):

        self.initialize_experiment()
        if pruning_value > 0 and self.pruning_function:
            self.pruning_function(self.model, pruning_value)

        validation_stats = self.validate_epoch()

        return validation_stats


########## Define pruning functions ##########
def pruning_method_1(model, pruning_factor):
    """
    Prune the second convolutional layer and the first
    linear layer in the output layer using unstructured pruning
    with L1 norm.
    """
    conv_layer = model.features[2]
    prune.l1_unstructured(conv_layer, name="weight", amount=pruning_factor)

    lin_layer = model.fc[0]
    prune.l1_unstructured(lin_layer, name="weight", amount=pruning_factor)


def pruning_method_2(model, pruning_factor):
    """
    Prune all feature extractor and the first linear layer
    unstructured pruning with L1 norm.
    """
    conv_layer1 = model.features[0]
    prune.l1_unstructured(conv_layer1, name="weight", amount=pruning_factor)

    conv_layer2 = model.features[2]
    prune.l1_unstructured(conv_layer2, name="weight", amount=pruning_factor)

    lin_layer = model.fc[0]
    prune.l1_unstructured(lin_layer, name="weight", amount=pruning_factor)


def pruning_method_3(model, pruning_factor):
    """
    Prune all layers using
    unstructured pruning with L1 norm.
    """
    conv_layer1 = model.features[0]
    prune.l1_unstructured(conv_layer1, name="weight", amount=pruning_factor)

    conv_layer2 = model.features[2]
    prune.l1_unstructured(conv_layer2, name="weight", amount=pruning_factor)

    lin_layer1 = model.fc[0]
    prune.l1_unstructured(lin_layer1, name="weight", amount=pruning_factor)

    lin_layer2 = model.fc[2]
    prune.l1_unstructured(lin_layer2, name="weight", amount=pruning_factor)


def pruning_method_4(model, pruning_factor):
    """
    Prune all feature extractor using structured pruning
    with the L2 norm along dim 0.
    """
    conv_layer1 = model.features[0]
    prune.ln_structured(conv_layer1, name="weight", amount=pruning_factor, n=2, dim=0)

    conv_layer2 = model.features[2]
    prune.ln_structured(conv_layer2, name="weight", amount=pruning_factor, n=2, dim=0)


### Experiment jobs
def create_baseline_experiment_result(logger, game, exp_out_folder, params_file_name):

    exp_out_file = os.path.join(exp_out_folder, "baseline")

    seed_everything(0)

    pruning_experiment = PruningExperiment(
        logger=logger,
        game=game,
        model_params_file_name=params_file_name,
        exp_out_file=exp_out_file,
        pruning_function=None,
        experiment_info="Baseline performance, no pruning was done.",
    )

    pruning_experiment.perform_single_experiment(0.00)


def run_experiment_with_params(params):
    # Extract individual parameters from the dictionary
    logger = setup_logger()
    game = params["game"]
    exp_out_folder = params["exp_out_folder"]
    exp_out_file_name = params["exp_out_file_name"]
    params_file_name = params["params_file_name"]
    pruning_function = params["pruning_function"]
    if "experiment_info" in params:
        experiment_info = params["experiment_info"]
    else:
        experiment_info = pruning_function.__doc__

    logger.info(f"Initializing experiment: {exp_out_file_name}")

    ### Setup and run pruning experiment
    exp_out_file = os.path.join(exp_out_folder, exp_out_file_name)

    seed_everything(0)

    pruning_experiment = PruningExperiment(
        logger=logger,
        game=game,
        model_params_file_name=params_file_name,
        exp_out_file=exp_out_file,
        pruning_function=pruning_function,
        experiment_info=experiment_info,
    )

    pruning_experiment.perform_multiple_experiments(
        [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    )

    return True


def run_parallel_pruning_experiment(logger, game, exp_out_folder, params_file_name):
    exp_1_params = {
        "game": game,
        "exp_out_folder": exp_out_folder,
        "exp_out_file_name": "pruning_results_1",
        "params_file_name": params_file_name,
        "pruning_function": pruning_method_1,
    }

    exp_2_params = {
        "game": game,
        "exp_out_folder": exp_out_folder,
        "exp_out_file_name": "pruning_results_2",
        "params_file_name": params_file_name,
        "pruning_function": pruning_method_2,
    }

    exp_3_params = {
        "game": game,
        "exp_out_folder": exp_out_folder,
        "exp_out_file_name": "pruning_results_3",
        "params_file_name": params_file_name,
        "pruning_function": pruning_method_3,
    }

    exp_4_params = {
        "game": game,
        "exp_out_folder": exp_out_folder,
        "exp_out_file_name": "pruning_results_4",
        "params_file_name": params_file_name,
        "pruning_function": pruning_method_4,
    }

    experiment_params = [exp_1_params, exp_2_params, exp_3_params, exp_4_params]

    # initializer=setup_logger
    with multiprocessing.Pool() as pool:
        statuses = list(pool.map(run_experiment_with_params, experiment_params))

    logger.info(f"Parallel pruning status: {str(statuses)}")


def main():
    """
    Look in the folder with trained model for trained models
    The output of the pruning experiment mirrors the folder structure of the training experiment
    There is one additional nesting level for the pruning method
    """


    seed_everything(0)
    logger = setup_logger()

    # Collect all paths to models in a specified folder
    training_outputs_folder_path = (
        r"D:\Work\PhD\minatar_work\experiments\training\outputs"
    )
    training_timestamp_folder = "2023_02_24-00_20_13"

    model_file_path_list = search_files_with_string(
        os.path.join(training_outputs_folder_path, training_timestamp_folder), "model"
    )

    # build and create paths to pruning experiment outputs
    pruning_outputs_folder_path = (
        r"D:\Work\PhD\minatar_work\experiments\pruning\outputs"
    )
    for model_path in model_file_path_list:
        if "breakout" in model_path:
            game = "breakout"
            # print(model_path)

            left_path, abs_path_experiment_model = split_path_at_substring(
                model_path, training_timestamp_folder
            )
            folder_structure, model_name = os.path.split(abs_path_experiment_model)
            path_to_pruning_experiment_folder = os.path.join(
                pruning_outputs_folder_path, training_timestamp_folder, folder_structure
            )
            Path(path_to_pruning_experiment_folder).mkdir(parents=True, exist_ok=True)
            
            create_baseline_experiment_result(logger, game, path_to_pruning_experiment_folder, model_path)

            run_parallel_pruning_experiment(logger, game, path_to_pruning_experiment_folder, model_path)


    handlers = logger.handlers[:]
    for handler in handlers:
        logger.removeHandler(handler)
        handler.close()


if __name__ == "__main__":
    main()
