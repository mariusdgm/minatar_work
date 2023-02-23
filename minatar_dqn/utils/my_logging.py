import os
import sys

import logging
import datetime

def setup_logger(env_name="default_game", folder_path=None):
    ### Setup logging for training ####
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    if not logger.handlers:
        # create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # create file handler
    if folder_path:
        time_stamp = datetime.datetime.now().strftime(r"%Y_%m_%d-%I_%M_%S_%p")
        log_file_name = f"{env_name}_{time_stamp}"
        log_file_path = os.path.join(folder_path, log_file_name)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def cleanup_file_handlers(experiment_logger=None):
    # Get all active loggers
    if experiment_logger:
        for handler in experiment_logger.handlers:
            handler.close()
            experiment_logger.removeHandler(handler)
    
    else:
        logger = logging.getLogger()
        for handler in logger.handlers:
            handler.close()
            logger.removeHandler(handler)
        