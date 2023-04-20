import os
import sys

import logging
import datetime

def setup_logger(env_name="default_game", folder_path=None, identifier_string=None):
    """
    Create and return a customized logger.

    Args: 
        env_name: identifier string to be used in the name of the file (if the log file is created).
        folder_path: folder where to save the log file. If None, no folder is created.
        identifier_string: an identifier string that will appear in each logged message.

    Returns:
        Logger: customized logger instance.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if identifier_string is None:
        formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
    else:
        formatter = logging.Formatter(
                f"%(asctime)s - %(name)s - %(levelname)s - {identifier_string} - %(message)s"
            )

    stdout_handler_exists = False
    for handler in logging.getLogger().handlers:
        if hasattr(handler, 'stream') and handler.stream == sys.stdout:
            stdout_handler_exists = True

    # If no handler exists, add one that writes to sys.stdout
    if not stdout_handler_exists:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # create file handler
    if folder_path:
        time_stamp = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
        log_file_name = f"{env_name}_{time_stamp}"
        log_file_path = os.path.join(folder_path, log_file_name)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def cleanup_file_handlers(experiment_logger=None):
    """Cleans up all handlers of experiment_logger logger instance.

    Args:
        experiment_logger (Logger, optional): If experiment_logger is None, then all loggers will be cleaned up.
        Defaults to None.
    """
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
        