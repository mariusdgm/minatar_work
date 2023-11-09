import os, sys


def get_dir_n_levels_up(path, n):
    # Go up n levels from the given path
    for _ in range(n):
        path = os.path.dirname(path)
    return path


proj_root = get_dir_n_levels_up(os.path.abspath(__file__), 4)
sys.path.append(proj_root)

import yaml
import itertools
from pathlib import Path
import datetime
import multiprocessing
import traceback
from typing import List, Dict, Tuple

from liftoff import parse_opts


from minatar_dqn.utils import my_logging
from minatar_dqn import my_dqn
from experiments.experiment_utils import seed_everything

# os.environ["OMP_NUM_THREADS"] = "2"


def run(opts: Dict) -> True:
    """Start a training experiment using input configuration.

    Args:
        opts (NameSpace): Configuration to use in the experiment.

    Returns:
        bool: Returns True on experment end.
    """

    try:

        config = vars(opts)
        
        seed_everything(config["seed"])

        logs_folder = os.path.join(config["out_dir"], "logs")
        Path(logs_folder).mkdir(parents=True, exist_ok=True)

        env_name = config["environment"]
        logger = my_logging.setup_logger(
            env_name=env_name,
            folder_path=logs_folder,
            identifier_string=config["title"],
        )
        logger.info(
            f'Starting up experiment: {config["title"]}, environment: {config["environment"]}, seed: {config["seed"]}'
        )

        # ### Setup environments ###
        # train_env = my_dqn.build_environment(
        #     game_name=config["environment"], random_seed=config["seed"]
        # )
        # validation_env = my_dqn.build_environment(
        #     game_name=config["environment"], random_seed=config["seed"]
        # )

        # ### Setup output and loading paths ###

        # path_previous_experiments_outputs = None
        # if "restart_training_timestamp" in config:
        #     path_previous_experiments_outputs = create_path_to_experiment_folder(
        #         config,
        #         path_experiments_outputs,
        #         config["restart_training_timestamp"],
        #     )

        # config["experiment_output_folder"] = exp_folder_path
        # config["full_experiment_name"] = experiment_file_string

        # config_to_record = os.path.join(exp_folder_path, f"{experiment_file_string}_config")
        # with open(config_to_record, "w") as file:
        #     yaml.dump(config, file)

        # experiment_agent = my_dqn.AgentDQN(
        #     train_env=train_env,
        #     validation_env=validation_env,
        #     experiment_output_folder=exp_folder_path,
        #     experiment_name=experiment_file_string,
        #     resume_training_path=path_previous_experiments_outputs,
        #     save_checkpoints=True,
        #     logger=logger,
        #     config=config,
        # )

        # experiment_agent.train(train_epochs=config["epochs_to_train"])

        logger.info(
            f'Finished training experiment: {config["experiment_name"]}, environment: {config["environment"]}, seed: {config["seed"]}'
        )

        my_logging.cleanup_file_handlers(experiment_logger=logger)

        return True

    except Exception as exc:
        # Capture the stack trace along with the exception message
        error_info = traceback.format_exc()

        # Log this information using your logger, if it's available
        if 'logger' in locals():
            logger.error("An error occurred: %s", error_info)

        # Return the error info so it can be collected by the parent process
        return error_info


### old implementation for parallelization
# def start_parallel_training_session(
#     configs: List[Dict],
#     restart_training_timestamp: str = None,
#     processes: int = 8
# ) -> None:
#     """Function call to start multiple training sessions in parallel.

#     Args:
#         configs (List[Dict]): list with the configurations to be used in the experiments.
#         restart_training_timestamp (str, optional): Datetime string that represents the folder name of a previous output.
#                                                     Defaults to None.
#         processes (int, optional): How many parallel processes to start. Defaults to 8.
#     """

#     # add this parameter to every training config so that we group them in the same training session.
#     current_timestamp = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
#     for conf in configs:
#         conf["experiment_start_timestamp"] = current_timestamp

#     if restart_training_timestamp:
#         for conf in configs:
#             conf["restart_training_timestamp"] = restart_training_timestamp

#     with multiprocessing.Pool(processes=processes) as pool:
#         results = list(pool.map(run, configs))

#     for config, result in zip(configs, results):
#         if result is not True:
#             print(f"Error in config {config['experiment_name']}: {result}")

#     print("All parallel jobs completed.")

# def start_single_training_session(
#     config: Dict,
#     restart_training_timestamp: str = None,
# ) -> None:
#     """Function call to start multiple training sessions in parallel.

#     Args:
#         configs (Dict): list with the configurations to be used in the experiments.
#         restart_training_timestamp (str, optional): Datetime string that represents the folder name of a previous output.
#                                                     Defaults to None.
#         processes (int, optional): How many parallel processes to start. Defaults to 8.
#     """

#     # add this parameter to every training config so that we group them in the same training session.
#     current_timestamp = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
#     config["experiment_start_timestamp"] = current_timestamp

#     if restart_training_timestamp:
#         config["restart_training_timestamp"] = restart_training_timestamp

#     run(config)

# def main():

#     seed_everything(0)

#     file_dir = os.path.dirname(os.path.abspath(__file__))
#     path_experiments_configs = os.path.join(file_dir, "training_configs")
#     path_experiments_outputs = os.path.join(file_dir, "outputs")

#     default_config_path, experiment_config_paths = get_config_paths(
#         path_experiments_configs
#     )

#     experiment_configs = read_config_files(default_config_path, experiment_config_paths)

#     runs_configs = generate_run_configs(experiment_configs, path_experiments_outputs)

#     start_parallel_training_session(
#         runs_configs
#     )

#     # print(runs_configs[1])
#     # start_single_training_session(runs_configs[1])

#     my_logging.cleanup_file_handlers()


### Liftoff implementation
def main():
    opts = parse_opts()
    run(opts)


if __name__ == "__main__":
    main()
