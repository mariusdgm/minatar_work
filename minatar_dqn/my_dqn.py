import time
import datetime
import torch
import random
import numpy as np
import os
from pathlib import Path
import argparse

import torch.optim as optim
import torch.nn.functional as F

from minatar import Environment

from minatar_dqn.replay_buffer import ReplayBuffer
from experiments.experiment_utils import seed_everything
from minatar_dqn.utils.my_logging import setup_logger
from minatar_dqn.models import Conv_QNET, Conv_QNET_one

# TODO: (NICE TO HAVE) gpu device at: model, wrapper of environment (in my case it would be get_state...),
# maybe: replay buffer (recommendation: keep on cpu, so that the env can run on gpu in parallel for multiple experiments)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

# TODO: Maybe implement this as wrapper for env?
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
        train_env,
        validation_env,
        output_files_paths=None,
        load_file_paths=None,
        save_checkpoints=True,
        logger=None,
        config=None,
    ) -> None:

        # assign environments
        self.train_env = train_env
        self.validation_env = validation_env

        # assign output files
        self.model_file = output_files_paths["model_file"]
        self.replay_buffer_file = output_files_paths["replay_buffer_file"]
        self.train_stats_file = output_files_paths["train_stats_file"]

        self.save_checkpoints = save_checkpoints
        self.logger = logger

        self._load_config_settings(config)

        self._init_models(config)  # init policy, target and optim

        # Set initial values related to training and monitoring
        self.t = 0  # frame nr
        self.episodes = 0  # episode nr
        self.policy_model_update_counter = 0

        self.reset_training_episode_tracker()

        self.training_stats = []
        self.validation_stats = []

        # check that all paths were provided and that the files can be found
        if load_file_paths:
            self._check_load_paths(load_file_paths)
            self.load_training_state(load_file_paths)

    def _check_load_paths(self, load_file_paths):
        expected_keys = ["model_file", "replay_buffer_file", "train_stats_file"]
        for file in expected_keys:
            if file not in load_file_paths:
                raise KeyError(f"Key {file} missing from load_file_paths argument.")
            if not os.path.exists(load_file_paths[file]):
                raise FileNotFoundError(
                    f"Could not find the file {load_file_paths[file]} for {file}."
                )

    def _load_config_settings(self, config={}):
        """
        Load the settings from config.
        If config was not provided, then default values are used.
        """
        agent_params = config.get("agent_params", {}).get("args_", {})

        # setup training configuration
        self.train_step_cnt = agent_params.get("train_step_cnt", 200_000)
        self.validation_enabled = agent_params.get("validation_enabled", True)
        self.validation_step_cnt = agent_params.get("validation_step_cnt", 100_000)
        self.validation_epsilon = agent_params.get("validation_epsilon", 0.001)

        self.replay_start_size = agent_params.get("replay_start_size", 5_000)

        self.batch_size = agent_params.get("batch_size", 32)
        self.training_freq = agent_params.get("training_freq", 4)
        self.target_model_update_freq = agent_params.get(
            "target_model_update_freq", 100
        )
        self.gamma = agent_params.get("gamma", 0.99)
        self.loss_function = agent_params.get("loss_fcn", "mse_loss")

        eps_settings = agent_params.get(
            "epsilon", {"start": 1.0, "end": 0.01, "decay": 250_000}
        )
        self.epsilon_by_frame = self._get_linear_decay_function(
            start=eps_settings["start"],
            end=eps_settings["end"],
            decay=eps_settings["decay"],
            eps_decay_start=self.replay_start_size,
        )

        self._read_and_init_envs()  # sets up in_features etc...

        buffer_settings = config.get(
            "replay_buffer", {"max_size": 100_000, "action_dim": 1, "n_step": 0}
        )
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_settings.get("max_size", 100_000),
            state_dim=self.in_features,
            action_dim=buffer_settings.get("action_dim", 1),
            n_step=buffer_settings.get("n_step", 0),
        )

        self.logger.info("Loaded configuration settings.")

    def _get_exp_decay_function(self, start, end, decay):
        return lambda x: end + (start - end) * np.exp(-1.0 * x / decay)

    def _get_linear_decay_function(self, start, end, decay, eps_decay_start=None):
        """Return a function that enables getting the value of epsilon at step x.

        Args:
            start (float): start value of the epsilon function (x=0)
            end (float): end value of the epsilon function (x=decay)
            decay (int): how many steps to reach the end value
            eps_decay_start: after how many frames to actually start decaying,
                            uses self.replay_start_size by default

        Returns:
            function to compute the epsillon based on current frame counter
        """
        if not eps_decay_start:
            eps_decay_start = self.replay_start_size

        return lambda x: max(
            end, min(start, start - (start - end) * ((x - eps_decay_start) / decay))
        )

    def _check_path(self, var_name, path):
        if path is None:
            raise ValueError("Provide a path")

    def _init_models(self, config):
        estimator_settings = config.get(
            "estimator", {"model": "Conv_QNET", "args_": {}}
        )

        if estimator_settings["model"] == "Conv_QNET":
            self.policy_model = Conv_QNET(
                self.in_features,
                self.in_channels,
                self.num_actions,
                **estimator_settings["args_"],
            )
            self.target_model = Conv_QNET(
                self.in_features,
                self.in_channels,
                self.num_actions,
                **estimator_settings["args_"],
            )
        elif estimator_settings["model"] == "Conv_QNET_one":
            self.policy_model = Conv_QNET_one(
                self.in_features,
                self.in_channels,
                self.num_actions,
                **estimator_settings["args_"],
            )
            self.target_model = Conv_QNET_one(
                self.in_features,
                self.in_channels,
                self.num_actions,
                **estimator_settings["args_"],
            )
        else:
            estiamtor_name = estimator_settings["model"]
            raise ValueError(f"Could not setup estimator. Tried with: {estiamtor_name}")

        optimizer_settings = config.get("optim", {"name": "Adam", "args_": {}})
        self.optimizer = optim.Adam(
            self.policy_model.parameters(), **optimizer_settings["args_"]
        )

        self.logger.info("Initialized newtworks and optimizer.")

    def _read_and_init_envs(self):
        """Read dimensions of the input and output of the simulation environment"""
        # returns state as [w, h, channels]
        state_shape = self.train_env.state_shape()

        # permute to get batch, channel, w, h shape
        # specific to minatar
        self.in_features = (state_shape[2], state_shape[0], state_shape[1])
        self.in_channels = self.in_features[0]
        self.num_actions = self.train_env.num_actions()

        self.train_env.reset()
        self.validation_env.reset()

    def load_training_state(self, load_file_paths):
        self.load_models(load_file_paths["model_file"])
        self.policy_model.train()
        self.target_model.train()
        self.load_training_stats(load_file_paths["train_stats_file"])
        self.replay_buffer.load(load_file_paths["replay_buffer_file"])

        self.logger.info(
            f"Loaded previous training status from the following files: {str(load_file_paths)}"
        )

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

        self.logger.info(
            f"Ended training session after {train_epochs} epochs at t = {self.t}"
        )

    def train_epoch(self):
        self.logger.info(f"Starting training epoch at t = {self.t}")
        epoch_t = 0
        policy_trained_times = 0
        target_trained_times = 0

        epoch_episode_rewards = []
        epoch_episode_nr_frames = []
        epoch_losses = []
        epoch_max_qs = []

        start_time = datetime.datetime.now()
        while epoch_t < self.train_step_cnt:
            (
                is_terminated,
                epoch_t,
                current_episode_reward,
                ep_frames,
                ep_policy_trained_times,
                ep_target_trained_times,
                ep_losses,
                ep_max_qs,
            ) = self.train_episode(epoch_t, self.train_step_cnt)

            policy_trained_times += ep_policy_trained_times
            target_trained_times += ep_target_trained_times

            if is_terminated:
                # we only want to append these stats if the episode was completed,
                # otherwise it means it was stopped due to the nr of frames criterion
                epoch_episode_rewards.append(current_episode_reward)
                epoch_episode_nr_frames.append(ep_frames)
                epoch_losses.extend(ep_losses)
                epoch_max_qs.extend(ep_max_qs)

                self.episodes += 1
                self.reset_training_episode_tracker()

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_training_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_nr_frames,
            policy_trained_times,
            target_trained_times,
            epoch_losses,
            epoch_max_qs,
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

        epoch_episode_rewards = []
        epoch_episode_nr_frames = []
        epoch_max_qs = []
        valiation_t = 0

        start_time = datetime.datetime.now()

        while valiation_t < self.validation_step_cnt:
            (
                current_episode_reward,
                ep_frames,
                ep_max_qs,
            ) = self.validate_episode()

            valiation_t += ep_frames

            epoch_episode_rewards.append(current_episode_reward)
            epoch_episode_nr_frames.append(ep_frames)
            epoch_max_qs.extend(ep_max_qs)

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_validation_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_nr_frames,
            epoch_max_qs,
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

    def validate_episode(self):

        current_episode_reward = 0.0
        ep_frames = 0
        max_qs = []

        # Initialize the environment and start state
        self.validation_env.reset()
        s = get_state(self.validation_env.state())

        is_terminated = False
        while not is_terminated:
            action, max_q = self.select_action(
                s, self.t, self.num_actions, epsilon=self.validation_epsilon
            )
            reward, is_terminated = self.validation_env.act(action)
            s_prime = get_state(self.validation_env.state())

            max_qs.append(max_q)

            current_episode_reward += reward

            ep_frames += 1

            # Continue the process
            s = s_prime

        return (
            current_episode_reward,
            ep_frames,
            max_qs,
        )

    def reset_training_episode_tracker(self):
        self.current_episode_reward = 0.0
        self.ep_frames = 0
        self.losses = []
        self.max_qs = []

        self.train_env.reset()

    def train_episode(self, epoch_t, train_frames):
        policy_trained_times = 0
        target_trained_times = 0

        s = get_state(self.train_env.state())

        is_terminated = False
        while (not is_terminated) and (
            epoch_t < train_frames
        ):  # can early stop episode if the frame limit was reached

            action, max_q = self.select_action(s, self.t, self.num_actions)
            reward, is_terminated = self.train_env.act(action)
            reward = torch.tensor([[reward]], device=device).float()
            is_terminated = torch.tensor([[is_terminated]], device=device)
            s_prime = get_state(self.train_env.state())

            self.replay_buffer.append(s, action, reward, s_prime, is_terminated)

            self.max_qs.append(max_q)

            # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
            if (
                self.t > self.replay_start_size
                and len(self.replay_buffer) >= self.batch_size
            ):
                # Train every training_freq number of frames
                if self.t % self.training_freq == 0:
                    sample = self.replay_buffer.sample(self.batch_size)
                    loss_val = self.model_learn(sample)

                    self.losses.append(loss_val)
                    self.policy_model_update_counter += 1
                    policy_trained_times += 1

                # Update the target network only after some number of policy network updates
                if (
                    self.policy_model_update_counter > 0
                    and self.policy_model_update_counter % self.target_model_update_freq
                    == 0
                ):
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                    target_trained_times += 1

            self.current_episode_reward += reward.item()

            self.t += 1
            epoch_t += 1
            self.ep_frames += 1

            # Continue the process
            s = s_prime

        # end of episode, return episode statistics:
        return (
            is_terminated,
            epoch_t,
            self.current_episode_reward,
            self.ep_frames,
            policy_trained_times,
            target_trained_times,
            self.losses,
            self.max_qs,
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

        if self.loss_function == "mse_loss":
            loss = F.mse_loss(selected_q_value, expected_q_value)

        # loss = F.smooth_l1_loss(q_value, expected_q_value)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# TODO: fix this because class was changed
def classic_experiment():
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

    train_env = build_environment(game_name=args.game, random_seed=0)
    validation_env = build_environment(game_name=args.game, random_seed=0)

    train_logger = setup_logger(args.game, logs_path)

    my_agent = AgentDQN(
        train_env=train_env,
        validation_env=validation_env,
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


def build_environment(game_name, random_seed):
    """Wrapper function for creating a simulation environment.

    Args:
        game_name: the name of the environment
        random_seed: seed to be used for initializaion

    Returns:
        Environment object that implements functions for step wise simulation and reward returns.

    For more information check MinAtar documentation
    """

    return Environment(game_name, random_seed)


def main():
    pass


if __name__ == "__main__":

    seed_everything(0)
    main()
    # play_game_visual("breakout")
