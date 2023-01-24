import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import torch

import pandas as pd


def read_training_logs(checkpoint_load_path):
    checkpoint = torch.load(checkpoint_load_path)

    training_stats = {}
    training_stats["avg_episode_rewards"] = checkpoint["avg_episode_rewards"]
    training_stats["avg_episode_nr_frames"] = checkpoint["avg_episode_nr_frames"]
    training_stats["log_frame_stamp"] = checkpoint["log_frame_stamp"]

    validation_stats = {}
    validation_stats["val_avg_episode_rewards"] = checkpoint["val_avg_episode_rewards"]
    validation_stats["val_avg_episode_nr_frames"] = checkpoint[
        "val_avg_episode_nr_frames"
    ]
    validation_stats["val_log_frame_stamp"] = checkpoint["val_avg_episode_nr_frames"]

    return training_stats, validation_stats


def plot_episodic_logs(fig, axs, ep_rewards, ep_frames, log_indx, title):
    fig.suptitle(title)
    fig.set_figheight(8)
    fig.set_figwidth(12)
    sns.lineplot(x=log_indx, y=ep_rewards, ax=axs[0])
    axs[0].set_ylabel(
        "Avg episode rewards",
    )
    axs[0].set_xlabel(
        "Frames",
    )

    sns.lineplot(x=log_indx, y=ep_frames, ax=axs[1])
    axs[1].set_ylabel(
        "Avg episode frames",
    )
    axs[1].set_xlabel(
        "Frames",
    )

def parse_log_example():
    log_data = pd.read_csv("training.log", sep=" - ", header=None, names=["time", "name", "level", "message"])

    # extract the relevant information
    episodes = []
    rewards = []
    for message in log_data["message"]:
        if "Episode" in message:
            episode = int(message.split()[1])
            reward = float(message.split()[-1])
            episodes.append(episode)
            rewards.append(reward)

if __name__ == "__main__":
    game = "breakout"
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_folder = os.path.join(proj_dir, game)
    load_file_path = os.path.join(default_save_folder, game + "_checkpoint")

    training_stats, validation_stats = read_training_logs(load_file_path)

    fig, axs = plt.subplots(nrows=2, ncols=1)

    plot_episodic_logs(
        fig,
        axs,
        training_stats["avg_episode_rewards"],
        training_stats["avg_episode_nr_frames"],
        training_stats["log_frame_stamp"],
        "Training stats",
    )
