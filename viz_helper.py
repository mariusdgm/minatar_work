import os

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import torch

import pandas as pd


def load_training_stats(training_stats_file):
    checkpoint = torch.load(training_stats_file)

    training_stats = checkpoint["training_stats"]
    validation_stats = checkpoint["validation_stats"]

    return training_stats, validation_stats

def get_df_of_stat(stats, stat_name):
    x_idx = []
    stat_records = []

    for ep_stats in stats:
        x_idx.append(ep_stats["frame_stamp"])
        stat_records.append(ep_stats[stat_name])
    
    df = pd.DataFrame.from_records(stat_records, index = x_idx) 
    df = df.reset_index()
    df = df.rename(columns = {'index': 'frames'})

    return df

def plot_stat_log(stats, stat_name, title):
    df = get_df_of_stat(stats, stat_name=stat_name)

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
    fig.suptitle(title)

    sns.lineplot(data=df, x="frames", y="mean", ax=axs[0])
    axs[0].set_ylabel(
        f"{stat_name} mean",
    )
    axs[0].set_xlabel(
        "Frames",
    )

    sns.lineplot(data=df, x="frames", y="median", ax=axs[1])
    axs[1].set_ylabel(
        f"{stat_name} median",
    )
    axs[1].set_xlabel(
        "Frames",
    )

    sns.lineplot(data=df, x="frames", y="max", ax=axs[2])
    axs[2].set_ylabel(
        f"{stat_name} max",
    )
    axs[2].set_xlabel(
        "Frames",
    )



if __name__ == "__main__":
    game = "breakout"
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_folder = os.path.join(proj_dir, "checkpoints", game)
    file_name = os.path.join(default_save_folder, game + "_train_stats")

    training_stats, validation_stats = load_training_stats(file_name)

    # plot_stat_log(training_stats, stat_name="episode_rewards", title="Episodic rewards")
    # plot_stat_log(training_stats, stat_name="episode_frames", title="Episodic length")
    # plot_stat_log(training_stats, stat_name="episode_losses", title="Training loss")
    # plot_stat_log(training_stats, stat_name="episode_max_qs", title="Episodic Q vals")
    
    plot_stat_log(validation_stats, stat_name="episode_rewards", title="Episodic rewards")
    plot_stat_log(validation_stats, stat_name="episode_frames", title="Episodic length")
    plot_stat_log(validation_stats, stat_name="episode_max_qs", title="Episodic Q vals")

    plt.show()