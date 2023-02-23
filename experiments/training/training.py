import os, sys
import yaml
import itertools
from pathlib import Path
import datetime

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

from minatar_dqn.utils import my_logging
from minatar_dqn import my_dqn
from experiments.experiment_setup import seed_everything

def get_config_paths(path_experiments_configs):
    experiment_config_paths = []
    default_config_path = None
    for root, dirs, files in os.walk(path_experiments_configs):
        for file in files:
            # check if the file has the specified extension
            if file.endswith("yaml"):
                if "default_params" in file:
                    default_config_path = os.path.join(path_experiments_configs, file)
                else:
                    experiment_config_paths.append(
                        os.path.join(path_experiments_configs, file)
                    )

    return default_config_path, experiment_config_paths


def read_config_files(default_config_path, experiment_config_paths):
    """
    Reads the contents of the configuration files and merges them
    with the default configuration.
    The settings in the specific experiments overwrite the settings
    in the default coinfiguration file.
    """
    default_config = None
    if default_config_path:
        with open(default_config_path, "r") as f:
            default_config = yaml.safe_load(f)

    experiment_configs = []
    for config_path in experiment_config_paths:
        # Load specific configuration from specific_config.yaml
        with open(config_path, "r") as f:
            specific_config = yaml.safe_load(f)

        if default_config:
            # Merge the experiment configuration with the default configuration
            config = {**default_config, **specific_config}
        else:
            config = specific_config

        # add original config file to configuration info
        experiment_file_name = os.path.basename(config_path)
        experiment_file_name = experiment_file_name.split(".")[0]
        config["experiment_name"] = experiment_file_name

        experiment_configs.append(config)

    return experiment_configs


def generate_run_configs(experiment_configs, path_experiments_outputs):
    """
    Generates the combination of configurations for each environment and seed
    specified in the training_configs.
    """
    runs_configs = []
    for experiment in experiment_configs:
        combinations = list(
            itertools.product(experiment["environments"], experiment["seeds"])
        )

        for env_seed_pair in combinations:
            single_experiment = {
                key: value
                for key, value in experiment.items()
                if key not in ["environments", "seed"]
            }
            # add base output path
            single_experiment["path_experiments_outputs"] = path_experiments_outputs
            single_experiment["environment"] = env_seed_pair[0]
            single_experiment["seed"] = env_seed_pair[1]
            runs_configs.append(single_experiment)

    return runs_configs


def create_exp_folder(config, experiments_output_folder, time_stamp=None):
    """
    Build the path for the nested experiment structure:
    base_outputs / timestamp / experiment / environment / seed

    Args:
        config: configuration of the experiment
        experiments_output_folder:
    """

    experiment = config["experiment_name"]
    env = config["environment"]
    seed = config["seed"]

    if time_stamp:
        # build path and check that it exists
        exp_folder_path = os.path.join(
            experiments_output_folder, time_stamp, experiment, env, str(seed)
        )
        if not os.path.exists(experiments_output_folder):
            raise ValueError(
                f"Could not find and existing path from a previous training run at: {experiments_output_folder}. \
                Check the value of the timestamp folder again."
            )

    else:
        # build path and create the folder
        time_stamp = datetime.datetime.now().strftime(r"%Y_%m_%d-%I_%M_%S_%p")
        exp_folder_path = os.path.join(
            experiments_output_folder, time_stamp, experiment, env, str(seed)
        )
        Path(exp_folder_path).mkdir(parents=True, exist_ok=True)

    return exp_folder_path


def get_training_file_names(exp_folder_path, experiment_file_string):

    model_file_name = os.path.join(exp_folder_path, experiment_file_string + "_model")
    replay_buffer_file = os.path.join(
        exp_folder_path, experiment_file_string + "_replay_buffer"
    )
    train_stats_file = os.path.join(
        exp_folder_path, experiment_file_string + "_train_stats"
    )

    return {
        "model_file": model_file_name,
        "replay_buffer_file": replay_buffer_file,
        "train_stats_file": train_stats_file,
    }


def run_training_experiment(config, restart_training_timestamp=None):

    path_experiments_outputs = config["path_experiments_outputs"]
    exp_folder_path = create_exp_folder(config, path_experiments_outputs)

    logs_folder = os.path.join(exp_folder_path, "logs")
    Path(logs_folder).mkdir(parents=True, exist_ok=True)

    env_name = config["environment"]
    logger = my_logging.setup_logger(env_name=env_name, folder_path=logs_folder)
    logger.info(
        f'Starting up experiment: {config["experiment_name"]}, environment: {config["environment"]}, seed: {config["seed"]}'
    )

    ### Setup environments ###
    train_env = my_dqn.build_environment(
        game_name=config["environment"], random_seed=config["seed"]
    )
    validation_env = my_dqn.build_environment(
        game_name=config["environment"], random_seed=config["seed"]
    )

    ### Setup output and loading paths ###
    experiment_file_string = (
        f'{config["experiment_name"]}_{config["environment"]}_{config["seed"]}'
    )

    output_files_paths = get_training_file_names(
        exp_folder_path, experiment_file_string
    )

    load_file_paths = None
    if restart_training_timestamp:
        path_previous_experiments_outputs = create_exp_folder(
            config, path_experiments_outputs, restart_training_timestamp
        )
        load_file_paths = get_training_file_names(
            path_previous_experiments_outputs, experiment_file_string
        )

    config["output_files_paths"] = output_files_paths
    config["load_file_paths"] = load_file_paths

    config_to_record = os.path.join(exp_folder_path, f"{experiment_file_string}_config")
    with open(config_to_record, "w") as file:
        yaml.dump(config, file)

    # experiment_agent = my_dqn.AgentDQN(
    #     train_env=train_env,
    #     validation_env=validation_env,
    #     output_files_paths=output_files_paths,
    #     load_file_paths=load_file_paths,
    #     save_checkpoints=True,
    #     logger=logger,
    #     config=config,
    # )
    # experiment_agent.train(train_epochs=config["epochs_to_train"])

    my_logging.cleanup_file_handlers(experiment_logger=logger)


def main():

    seed_everything(0)

    file_dir = os.path.dirname(os.path.abspath(__file__))
    path_experiments_configs = os.path.join(file_dir, "training_configs")
    path_experiments_outputs = os.path.join(file_dir, "outputs")

    default_config_path, experiment_config_paths = get_config_paths(
        path_experiments_configs
    )

    experiment_configs = read_config_files(default_config_path, experiment_config_paths)

    runs_configs = generate_run_configs(experiment_configs, path_experiments_outputs)

    for i, config in enumerate(runs_configs):

        # print(config)
        # print(i)

        run_training_experiment(config)


    my_logging.cleanup_file_handlers()


if __name__ == "__main__":
    main()
