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


def get_df_of_stat(stats, stat_name, show_epochs=False, epoch_frames=200_000):
    frame_stamps = []
    stat_records = []

    for epoch_stats in stats:
        frame_stamps.append(epoch_stats["frame_stamp"])
        stat_records.append(epoch_stats[stat_name])

    df = pd.DataFrame.from_records(stat_records, index=frame_stamps)
    df = df.reset_index()

    if show_epochs:
        df = df.rename(columns={"index": "epoch"})
        df["epoch"] = df["epoch"] / epoch_frames
    else:
        df = df.rename(columns={"index": "frames"})

    return df


def get_df_of_stat_pruning(stats, stat_name):
    x_idx = []
    stat_records = []

    for pruning_val in stats:
        x_idx.append(pruning_val)
        stat_records.append(stats[pruning_val][stat_name])

    df = pd.DataFrame.from_records(stat_records, index=x_idx)
    df = df.reset_index()
    df = df.rename(columns={"index": "pruning_factor"})

    return df


def plot_stat_log(stats, stat_name, title, show_epochs=False):
    df = get_df_of_stat(stats, stat_name=stat_name, show_epochs=show_epochs)

    x_label = "frames"
    if show_epochs:
        x_label = "epoch"

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
    fig.suptitle(title)

    sns.lineplot(data=df, x=x_label, y="mean", ax=axs[0])
    axs[0].set_ylabel(
        f"{stat_name} mean",
    )
    axs[0].set_xlabel(
        x_label,
    )

    sns.lineplot(data=df, x=x_label, y="median", ax=axs[1])
    axs[1].set_ylabel(
        f"{stat_name} median",
    )
    axs[1].set_xlabel(
        x_label,
    )

    sns.lineplot(data=df, x=x_label, y="max", ax=axs[2])
    axs[2].set_ylabel(
        f"{stat_name} max",
    )
    axs[2].set_xlabel(
        x_label,
    )


def plot_training_info(train_log_file_name, show_epochs=False):
    training_stats, validation_stats = load_training_stats(train_log_file_name)

    # plot_stat_log(training_stats, stat_name="episode_rewards", title="Episodic rewards")
    # plot_stat_log(training_stats, stat_name="episode_frames", title="Episodic length")
    # plot_stat_log(training_stats, stat_name="episode_losses", title="Training loss")
    # plot_stat_log(training_stats, stat_name="episode_max_qs", title="Episodic Q vals")

    plot_stat_log(
        validation_stats,
        stat_name="episode_rewards",
        title="Episodic rewards",
        show_epochs=show_epochs,
    )
    plot_stat_log(
        validation_stats,
        stat_name="episode_frames",
        title="Episodic length",
        show_epochs=show_epochs,
    )
    plot_stat_log(
        validation_stats,
        stat_name="episode_max_qs",
        title="Episodic Q vals",
        show_epochs=show_epochs,
    )


def load_pruning_experiment_data(pruning_exp_file):
    """TODO"""
    checkpoint = torch.load(pruning_exp_file)
    pruning_stats = checkpoint["pruning_validation_results"]
    experiment_info = checkpoint["experiment_info"]
    experiment_info = experiment_info.replace("\n", "")
    experiment_info = " ".join(experiment_info.split())

    return pruning_stats, experiment_info


def get_df_of_pruning_stats(stats, stat_name):
    """Transform dict of experimental stats to dataframe.

    Args:
        stats: dict with statistics of experiment with the following structure:
               pruning_value: dict with episodic rewards stats, episodic length stats etc.
               The stats are the mean, median, std, min and max of the vector of values
               that was generated during the experiment.
        stat_name: which statistics to plot (ex: 'episode_rewards', 'episode_rewards')

    Returns:
        Pandas Dataframe with the statistic of interest for each pruning value in the experiment
    """
    x_idx = []
    stat_records = []

    for pruning_val in stats:
        x_idx.append(pruning_val)
        stat_records.append(stats[pruning_val][stat_name])

    df = pd.DataFrame.from_records(stat_records, index=x_idx)
    df = df.reset_index()
    df = df.rename(columns={"index": "pruning_factor"})

    return df


def plot_pruning_stat(stats, stat_name, title=None, show=False, plot_min_max=False):

    df = get_df_of_stat_pruning(stats, stat_name=stat_name)

    sns.catplot(x="pruning_factor", y="mean", kind="box", data=df, showfliers=False)

    std_label = "Standard Deviation"
    minmax_label = "Minimum/Maximum"
    median_label = "Median"
    mean_label = "Mean"

    plt.errorbar(
        x=df.index,
        y=df["mean"],
        yerr=df["std"],
        fmt="none",
        ecolor="black",
        elinewidth=3,
        label=std_label,
    )

    for i in df.index:
        plt.vlines(
            x=i,
            ymin=df.loc[i, "min"],
            ymax=df.loc[i, "mean"],
            color="black",
            linestyle="--",
        )
        plt.vlines(
            x=i,
            ymin=df.loc[i, "mean"],
            ymax=df.loc[i, "max"],
            color="black",
            linestyle="--",
        )
        plt.hlines(
            y=df.loc[i, "mean"],
            xmin=i - 0.3,
            xmax=i + 0.3,
            color="black",
            linewidth=1.5,
            label=median_label,
        )
        plt.hlines(
            y=df.loc[i, "median"],
            xmin=i - 0.3,
            xmax=i + 0.3,
            color="red",
            linewidth=1,
            label=mean_label,
        )
        plt.hlines(
            y=df.loc[i, "min"],
            xmin=i - 0.2,
            xmax=i + 0.2,
            color="blue",
            linestyle="--",
        )
        plt.hlines(
            y=df.loc[i, "max"],
            xmin=i - 0.2,
            xmax=i + 0.2,
            color="blue",
            linestyle="--",
        )

    handles = [
        plt.errorbar([], [], label=std_label),
        plt.hlines([], [], [], color="red", linewidth=1, label=median_label),
        plt.hlines([], [], [], color="black", linewidth=1, label=mean_label),
        plt.vlines([], [], [], color="blue", linestyle="--", label=minmax_label),
    ]
    plt.legend(handles=handles, loc="upper right")

    plt.title(title)

    if show:
        plt.show()


# TODO: also make function for comparison at same pruning value


def plot_pruning_experiment_data(baseline_log_file_name, pruning_log_file_name):

    # load baseline
    baseline_pruning_stats, baseline_exp_info = load_pruning_experiment_data(
        baseline_log_file_name
    )

    # load experiment
    pruning_stats, exp_info = load_pruning_experiment_data(pruning_log_file_name)

    # add baseline stats to experiment stats for comparison:
    for key in baseline_pruning_stats:
        pruning_stats[key] = baseline_pruning_stats[key]
    pruning_stats = {k: pruning_stats[k] for k in sorted(pruning_stats)}

    print(exp_info)

    plot_pruning_stat(pruning_stats, "episode_rewards", "Episodic rewards")
    plot_pruning_stat(pruning_stats, "episode_frames", "Episodic length")
    plot_pruning_stat(pruning_stats, "episode_max_qs", "Episodic max q val")

    plt.show()


if __name__ == "__main__":

    experiment_folder = r"D:\Work\PhD\minatar_work\experiments\training\outputs\2023_02_24-15_43_43\conv_model_16\breakout\0"

    train_log_file_name = os.path.join(
        experiment_folder, "conv_model_16_breakout_0_train_stats"
    )

    plot_training_info(train_log_file_name, show_epochs=True)

    # default_save_folder = r"D:\Work\PhD\minatar_work\checkpoints\breakout"

    # baseline_file_path = os.path.join(default_save_folder, "pruning_exp", "baseline")
    # pruning_log_file_name = os.path.join(
    #     default_save_folder, "pruning_exp", "pruning_results_1"
    # )

    # plot_pruning_experiment_data(baseline_file_path, pruning_log_file_name)

    plt.show()
