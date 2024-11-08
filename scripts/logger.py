import os
import sys
import logging


def setup_logger():
    '''
    This funtion is used to setup logging
    '''

    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_file_info = os.path.join(log_dir, 'Info.log')
    log_file_error = os.path.join(log_dir, 'Error.log')

    formatter = logging.Formatter('%(asctime)s - %(levelname)s :: %(message)s',
                                datefmt='%Y-%m-%d %H:%M')

    info_handler = logging.FileHandler(log_file_info)
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)

    error_handler = logging.FileHandler(log_file_error)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(info_handler)
    logger.addHandler(error_handler)    
    logger.addHandler(console_handler)

    return logger     
    

LOGGER = setup_logger()    