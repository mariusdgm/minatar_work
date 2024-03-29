import os, sys

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

import torch

import pandas as pd


def load_training_stats(training_stats_file):
    """Read a training status file and retrieve the training stats.

    Args:
        training_stats_file (string): A file created during the training of an agent.
        Contains information needed to resume training at that point.

    Returns:
        Tuple[dict, dict]: Tuple with the statistics contained in dictionaries.
                        The first element is the statistics of the training epochs.
                        The second element is the statistics of thestats of the validation epochs.
    """
    checkpoint = torch.load(training_stats_file)

    training_stats = checkpoint["training_stats"]
    validation_stats = checkpoint["validation_stats"]

    return training_stats, validation_stats


def get_df_of_stat(
    stats, stat_name, show_epochs=False, epoch_frames=200_000, experiment=None
):
    """Processes a training stats dictionary into a pd.Dataframe for easier data analysis.

    Args:
        stats (dict): dictionary with the epoch statistics
        stat_name (string): name of the statistic to be extracted. Examples: 'episode_rewards',
                            'episode_frames', 'episode_losses', 'episode_max_qs'.
        show_epochs (bool, optional): The 'index' of the record is saved as the number of the frame.
                                    If 'show_epochs' is True, then the 'index' is transformed to match the epoch number. Defaults to False.
        epoch_frames (int, optional): How many frames are expected to be in an epoch. Defaults to 200_000.
        experiment (string, optional): An identifier string to add to the dataframe as a column.
                                    Used to represent different experiments. Defaults to None.

    Returns:
        pd.Dataframe: the statistics represented as a dataframe.
    """
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

    if experiment:
        df["experiment"] = experiment

    return df


def get_df_of_stat_pruning(stats, stat_name):
    """Processes a pruning stats dictionary into a pd.Dataframe for easier data analysis.

    Args:
        stats (dict): dictionary with the pruning validation statistics
        stat_name (string): name of the statistic to be extracted. Examples: 'episode_rewards',
        'episode_frames', 'episode_losses', 'episode_max_qs'

    Returns:
        _type_: _description_
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


def plot_stat_log(stats, stat_name, title, show_epochs=False):
    """Plots the mean, median and max of a specific statistic in subplots.

    Args:
        stats (dict): dictionary with the epoch statistics
        stat_name (string): name of the statistic to be extracted. Examples: 'episode_rewards',
                            'episode_frames', 'episode_losses', 'episode_max_qs'.
        title (string): title to show on plot. Defaults to None.
        show_epochs (bool, optional): The 'index' of the record is saved as the number of the frame.
                                    If 'show_epochs' is True, then the 'index' is transformed to match the epoch number. Defaults to False.
    """

    x_label = "frames"
    if show_epochs:
        x_label = "epoch"

    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(12, 8))
    fig.suptitle(title)

    sns.lineplot(data=stats, x=x_label, y="mean", ax=axs[0])
    axs[0].set_ylabel(
        f"{stat_name} mean",
    )
    axs[0].set_xlabel(
        x_label,
    )

    sns.lineplot(data=stats, x=x_label, y="median", ax=axs[1])
    axs[1].set_ylabel(
        f"{stat_name} median",
    )
    axs[1].set_xlabel(
        x_label,
    )

    sns.lineplot(data=stats, x=x_label, y="max", ax=axs[2])
    axs[2].set_ylabel(
        f"{stat_name} max",
    )
    axs[2].set_xlabel(
        x_label,
    )


def plot_stat_log_multi(df, stat_name, title, show_epochs=False):
    """Plot a specific statistic of a training/validation session

    Args:
        df (pd.Dataframe): pd.Dataframe that contains the statistics per epoch
        stat_name (string): name of the statistic to be extracted. Examples: 'episode_rewards',
                            'episode_frames', 'episode_losses', 'episode_max_qs'.
        title (string): title to show on the plot
        show_epochs (bool, optional): The 'index' of the record is saved as the number of the frame.
                                    If 'show_epochs' is True, then the 'index' is transformed to match the epoch number. Defaults to False.
    """

    x_label = "frames"
    if show_epochs:
        x_label = "epoch"

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(12, 8))
    fig.suptitle(title)

    experiments = df["experiment"].unique()
    colors = sns.color_palette(n_colors=len(experiments))

    for i, exp in enumerate(experiments):
        exp_df = df[df["experiment"] == exp]
        sns.lineplot(
            data=exp_df, x=x_label, y="mean", ax=axs[0], color=colors[i], label=exp
        )
        sns.lineplot(
            data=exp_df, x=x_label, y="median", ax=axs[1], color=colors[i], label=exp
        )
        # sns.lineplot(
        #     data=exp_df, x=x_label, y="max", ax=axs[2], color=colors[i], label=exp
        # )

    axs[0].set_ylabel(
        f"{stat_name} mean",
    )
    axs[0].set_xlabel(
        x_label,
    )

    axs[1].set_ylabel(
        f"{stat_name} median",
    )
    axs[1].set_xlabel(
        x_label,
    )

    # axs[2].set_ylabel(
    #     f"{stat_name} max",
    # )
    # axs[2].set_xlabel(
    #     x_label,
    # )

    plt.legend()


def subplots_stats(train_log_file_name, stat_name, show_epochs):
    """Read, preprocess and plot a specific statistic of a training session.

    Args:
        train_log_file_name (string): Path to the training stats file.
        stat_name (string): name of the statistic to be extracted. Examples: 'episode_rewards',
                            'episode_frames', 'episode_losses', 'episode_max_qs'.
        show_epochs (bool, optional): The 'index' of the record is saved as the number of the frame.
                                    If 'show_epochs' is True, then the 'index' is transformed to match the epoch number. Defaults to False.
    """
    df = None
    for log_name in train_log_file_name:
        training_stats, validation_stats = load_training_stats(log_name)
        exp_file_name = os.path.basename(log_name)

        df_stats = get_df_of_stat(
            validation_stats,
            stat_name=stat_name,
            show_epochs=show_epochs,
            experiment=exp_file_name,
        )

        if df is None:
            df = df_stats
        else:
            df = pd.concat([df, df_stats], ignore_index=True)

    plot_stat_log_multi(
        df,
        stat_name=stat_name,
        title=stat_name,
        show_epochs=show_epochs,
    )


def plot_training_info(train_log_file_name, show_epochs=False):
    """Create plots of multiple statistics for a training session

    Args:
        train_log_file_name (string): Path to the training stats file.
        show_epochs (bool, optional): The 'index' of the record is saved as the number of the frame.
                                    If 'show_epochs' is True, then the 'index' is transformed to match the epoch number. Defaults to False.
    """
    if type(train_log_file_name) is list:
        subplots_stats(train_log_file_name, "episode_rewards", show_epochs)
        subplots_stats(train_log_file_name, "episode_frames", show_epochs)

    else:
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
    """Read the statistics of a pruning experiment

    Args:
        pruning_exp_file (string): Path to the statistics of the pruning experiment.

    Returns:
        Tuple[dict, string]: Returns 2 variables. The first one is
        a dictionary containing the statistics of a pruning experiment
        (similar to the trianing session stats).
        The second argument is a string describing the pruning experiment.
    """

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
    """Create the plot for a specific statistic of a pruning session.

    Args:
        stats (dict): dictionary containing the statistics
        stat_name (string): name of the statistic to be extracted. Examples: 'episode_rewards',
        'episode_frames', 'episode_losses', 'episode_max_qs'
        title (string): title to show on plot. Defaults to None.
        show (bool, optional): Wether to call plt.show inside the function. Defaults to False.
        plot_min_max (bool, optional): Wether to plot the minimum and maximums on the plot. Defaults to False.
    """
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
        if plot_min_max:
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
    ]

    if plot_min_max:
        handles.append(
            plt.vlines([], [], [], color="blue", linestyle="--", label=minmax_label)
        )

    plt.legend(handles=handles, loc="upper right")

    plt.title(title)

    if show:
        plt.show()


def plot_pruning_experiment_data(baseline_log_file_name, pruning_log_file_name):
    """Main call for the plotting of a pruning experiment.

    Args:
        baseline_log_file_name (string): Path to the file with the stats of the baseline validation epoch.
        pruning_log_file_name (string): Path to the file with the stats of the pruning validation epochs.
    """
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
    # plot_pruning_stat(pruning_stats, "episode_frames", "Episodic length")
    # plot_pruning_stat(pruning_stats, "episode_max_qs", "Episodic max q val")

    plt.show()


if __name__ == "__main__":
    base_path = r"D:\Work\repos\RL\minatar_work\experiments\pruning\outputs"
    timestamp_str = "2023_03_17-02_50_37"
    # model_str = "conv_model_16"
    model_str = "conv_model_32"
    # env = "breakout"
    # env = "asterix"
    env = "space_invaders"
    # env = "seaquest"

    seed = "0"

    pruning_method = "pruning_results_1"

    default_save_folder = os.path.join(base_path, timestamp_str, model_str, env, seed)
    baseline_file_path = os.path.join(default_save_folder, "baseline")
    pruning_log_file_name = os.path.join(default_save_folder, pruning_method)
    plot_pruning_experiment_data(baseline_file_path, pruning_log_file_name)

    # ##### Plot multiple experiments

    # build list with paths to stats files

    # training_outputs_folder_path = (
    #     r"D:\Work\repos\RL\minatar_work\experiments\training\outputs"
    # )

    # # training_timestamp_folder = "2023_03_16-14_45_10"

    # training_timestamp_folder = "2023_03_17-02_50_37"

    # model_file_path_list = search_files_ending_with_string(
    #     os.path.join(training_outputs_folder_path, training_timestamp_folder), "stats"
    # )

    # # game = "space_invaders"
    # # game = "breakout"
    # game = "seaquest"
    # # game = "asterix"

    # model_file_path_list = [file for file in model_file_path_list if game in file]
    # # print(model_file_path_list)

    # plot_training_info(model_file_path_list, show_epochs=True)

    # plt.show()
