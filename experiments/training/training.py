import os, sys
import yaml
import itertools
from pathlib import Path
import datetime
import multiprocessing
import traceback
from typing import List, Dict, Tuple

proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(proj_root)

from minatar_dqn.utils import my_logging
from minatar_dqn import my_dqn
from experiments.experiment_utils import seed_everything

os.environ["OMP_NUM_THREADS"] = "2"


def get_config_paths(path_experiments_configs: str):
    """Return the path to the default configuration and a list of specific experiment configurations.

    Args:
        path_experiments_configs (str): path where the configs can be found.

    Returns:
        Tuple[str, List]: The first element is the path to the default configuration file.
                            The second element is a list with the paths to specific configuration files.
    """
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


def read_config_files(
    default_config_path: str, experiment_config_paths: List[str]
) -> List[Dict]:
    """Reads the contents of the configuration files and merges them
    with the default configuration.

    The settings in the specific experiments overwrite the settings
    in the default coinfiguration file. Usually, the specific experiment configs only change a few
    parameters of the default config.

    Args:
        default_config_path (str): Path of the default configuration for training experiments.
        experiment_config_paths (List[str]): List that contains the paths to specific experiment config files.

    Returns:
        List[Dict]: List of the overwritten experiment configurations.
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


def generate_run_configs(
    experiment_configs: List[Dict], path_experiments_outputs: str
) -> List[Dict]:
    """Generates the configurations for each combination of environment and seed
    specified in the training configs.

    Args:
        experiment_configs (List): List of dicts that contain the settings of the training experiments.
        path_experiments_outputs (str): Base path to where the outputs of the training experiments should be saved.

    Returns:
        List[Dict]: A list with the settings for a specific, singular training experiment (experiment settings + environment + seed)
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
                if key not in ["environments", "seeds"]
            }
            # add base output path
            single_experiment["path_experiments_outputs"] = path_experiments_outputs
            single_experiment["environment"] = env_seed_pair[0]
            single_experiment["seed"] = env_seed_pair[1]
            runs_configs.append(single_experiment)

    return runs_configs


def create_path_to_experiment_folder(
    config: Dict,
    experiments_output_folder: str,
    timestamp_folder: str = None,
) -> str:
    """Build the path for the nested experiment structure:
    base_outputs / timestamp / experiment / environment / seed

    Args:
        config (Dict): Configuration of the experiment.
        experiments_output_folder (str): Root path for the folder where the outputs
                                        of paralelized experiments are stored.
        timestamp_folder (str, optional): Path to the previous top level output folder. If None, then a new top level folder
                                        is created with a string matching the current time. Defaults to None.

    Returns:
        str: The path to the folder that stores the output for this singular experiment
    """
    experiment = config["experiment_name"]
    env = config["environment"]
    seed = config["seed"]

    prev_output_not_expected = True

    if timestamp_folder is None:
        timestamp_folder = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
        prev_output_not_expected = False # disable creation of prev folder 

    exp_folder_path = os.path.join(
        experiments_output_folder,
        timestamp_folder,
        experiment,
        env,
        str(seed),
    )

    if prev_output_not_expected:
        Path(exp_folder_path).mkdir(parents=True, exist_ok=True)

    return exp_folder_path


def get_training_file_names(exp_folder_path: str, experiment_file_string: str) -> Dict:
    """Build the paths to the files that represent the training output using the standard
    naming convention of the project.

    Args:
        exp_folder_path (str): Path to the specific experiment.
        experiment_file_string (str): String used to identify the specific experiment.

    Returns:
        Dict: dictionary with the 3 files relevant for training: model parameters, replay buffer and training statistics.
    """

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


def run_training_experiment(config: Dict) -> True:
    """Start a training experiment using input configuration.

    Args:
        config (Dict): Configuration to use in the experiment.

    Returns:
        bool: Returns True on experment end.
    """

    try:
        seed_everything(config["seed"])

        path_experiments_outputs = config["path_experiments_outputs"]
        exp_folder_path = create_path_to_experiment_folder(
            config,
            path_experiments_outputs,
            timestamp_folder=config["experiment_start_timestamp"],
        )

        experiment_file_string = (
            f'{config["experiment_name"]}_{config["environment"]}_{config["seed"]}'
        )

        logs_folder = os.path.join(exp_folder_path, "logs")
        Path(logs_folder).mkdir(parents=True, exist_ok=True)

        env_name = config["environment"]
        logger = my_logging.setup_logger(
            env_name=env_name,
            folder_path=logs_folder,
            identifier_string=experiment_file_string,
        )
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

        path_previous_experiments_outputs = None
        if "restart_training_timestamp" in config:
            path_previous_experiments_outputs = create_path_to_experiment_folder(
                config,
                path_experiments_outputs,
                config["restart_training_timestamp"],
            )

        config["experiment_output_folder"] = exp_folder_path
        config["full_experiment_name"] = experiment_file_string

        config_to_record = os.path.join(exp_folder_path, f"{experiment_file_string}_config")
        with open(config_to_record, "w") as file:
            yaml.dump(config, file)
        
        experiment_agent = my_dqn.AgentDQN(
            train_env=train_env,
            validation_env=validation_env,
            experiment_output_folder=exp_folder_path,
            experiment_name=experiment_file_string,
            resume_training_path=path_previous_experiments_outputs,
            save_checkpoints=True,
            logger=logger,
            config=config,
        )

        experiment_agent.train(train_epochs=config["epochs_to_train"])

        logger.info(
            f'Finished training experiment: {config["experiment_name"]}, environment: {config["environment"]}, seed: {config["seed"]}'
        )

        my_logging.cleanup_file_handlers(experiment_logger=logger)

        return True

    except Exception as exc:
        return str(exc)



def start_parallel_training_session(
    configs: List[Dict],
    restart_training_timestamp: str = None,
    processes: int = 8
) -> None:
    """Function call to start multiple training sessions in parallel.

    Args:
        configs (List[Dict]): list with the configurations to be used in the experiments.
        restart_training_timestamp (str, optional): Datetime string that represents the folder name of a previous output.
                                                    Defaults to None.
        processes (int, optional): How many parallel processes to start. Defaults to 8.
    """
    
    # add this parameter to every training config so that we group them in the same training session.
    current_timestamp = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
    for conf in configs:
        conf["experiment_start_timestamp"] = current_timestamp

    if restart_training_timestamp:
        for conf in configs:
            conf["restart_training_timestamp"] = restart_training_timestamp

    with multiprocessing.Pool(processes=processes) as pool:
        statuses = list(pool.map(run_training_experiment, configs))

    print(f"Parallel job run statuses: {statuses}")

def start_single_training_session(
    config: Dict,
    restart_training_timestamp: str = None,
) -> None:
    """Function call to start multiple training sessions in parallel.

    Args:
        configs (Dict): list with the configurations to be used in the experiments.
        restart_training_timestamp (str, optional): Datetime string that represents the folder name of a previous output.
                                                    Defaults to None.
        processes (int, optional): How many parallel processes to start. Defaults to 8.
    """
    
    # add this parameter to every training config so that we group them in the same training session.
    current_timestamp = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
    config["experiment_start_timestamp"] = current_timestamp

    if restart_training_timestamp:
        config["restart_training_timestamp"] = restart_training_timestamp

    run_training_experiment(config)


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

    start_parallel_training_session(
        runs_configs
    )

    # start_single_training_session(runs_configs[4])

    my_logging.cleanup_file_handlers()


if __name__ == "__main__":
    main()
