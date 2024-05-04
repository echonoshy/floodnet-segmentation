import logging
import os

def setup_logging(logger_name="training"):
    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)  # Set the logging level

    # Create file handler which logs even debug messages
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file_handler = logging.FileHandler(f'logs/{logger_name}.log')
    file_handler.setLevel(logging.INFO)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s:  %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
