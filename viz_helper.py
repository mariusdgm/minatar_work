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

    for pruning_val in stats:
        x_idx.append(pruning_val)
        stat_records.append(stats[pruning_val][stat_name])

    df = pd.DataFrame.from_records(stat_records, index = x_idx) 
    df = df.reset_index()
    df = df.rename(columns = {'index': 'pruning_factor'})

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

def plot_training_info(train_log_file_name):
    training_stats, validation_stats = load_training_stats(train_log_file_name)

    # plot_stat_log(training_stats, stat_name="episode_rewards", title="Episodic rewards")
    # plot_stat_log(training_stats, stat_name="episode_frames", title="Episodic length")
    # plot_stat_log(training_stats, stat_name="episode_losses", title="Training loss")
    # plot_stat_log(training_stats, stat_name="episode_max_qs", title="Episodic Q vals")
    
    plot_stat_log(validation_stats, stat_name="episode_rewards", title="Episodic rewards")
    plot_stat_log(validation_stats, stat_name="episode_frames", title="Episodic length")
    plot_stat_log(validation_stats, stat_name="episode_max_qs", title="Episodic Q vals")


def load_pruning_experiment_data(pruning_exp_file):
    """TODO"""
    checkpoint = torch.load(pruning_exp_file)
    pruning_stats = checkpoint["pruning_validation_results"]

    return pruning_stats # reshaped data

def get_df_of_pruning_stats(stats, stat_name):
    x_idx = []
    stat_records = []

    for pruning_val in stats:
        x_idx.append(pruning_val)
        stat_records.append(stats[pruning_val][stat_name])

    df = pd.DataFrame.from_records(stat_records, index = x_idx) 
    df = df.reset_index()
    df = df.rename(columns = {'index': 'pruning_factor'})

    return df

def plot_pruning_stat(stats, stat_name, title=None):

    df = get_df_of_stat(stats, stat_name=stat_name)

    sns.catplot(x="pruning_factor", y="mean", kind="box", data=df, showfliers=False )

    plt.errorbar(x=df.index, y=df['mean'], yerr=df['std'], fmt='none', ecolor='black', elinewidth=3)

    plt.plot(df.index, df['min'], 'ro', markersize=4)
    plt.plot(df.index, df['max'], 'ro', markersize=4)

    for i in df.index:
        plt.vlines(x=i, ymin=df.loc[i, 'min'], ymax=df.loc[i, 'mean'], color='black', linestyle='--')
        plt.vlines(x=i, ymin=df.loc[i, 'mean'], ymax=df.loc[i, 'max'], color='black', linestyle='--')
        plt.hlines(y=df.loc[i, 'mean'], xmin=i-0.3, xmax=i+0.3, color='black', linewidth=1.5)
        plt.hlines(y=df.loc[i, 'median'], xmin=i-0.3, xmax=i+0.3, color='red', linewidth=1)

    plt.title(title)

    plt.show()
    

def plot_pruning_experiment_data(pruning_log_file_name):
    
    pruning_stats = load_pruning_experiment_data(pruning_log_file_name)

    plot_pruning_stat(pruning_stats, "episode_rewards", "Episodic rewards")
    plot_pruning_stat(pruning_stats, "episode_frames", "Episodic length")
    plot_pruning_stat(pruning_stats, "episode_max_qs", "Episodic max q val")


if __name__ == "__main__":
    game = "breakout"
    proj_dir = os.path.dirname(os.path.abspath(__file__))
    default_save_folder = os.path.join(proj_dir, "checkpoints", game)
    train_log_file_name = os.path.join(default_save_folder, game + "_train_stats")
    
    pruning_log_file_name = os.path.join(default_save_folder, "pruning_exp", "pruning_results")

    # plot_training_info(train_log_file_name)

    plot_pruning_experiment_data(pruning_log_file_name)
    

    plt.show()