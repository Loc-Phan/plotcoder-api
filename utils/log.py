import sys
from loguru import logger


class StreamToLogger:
    def __init__(self, log_level='INFO'):
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def setup_logger(
    log_file_path: str, level: str = 'INFO', rotation=None, retention=None
):
    """logger setting
    Args:
        log_file_path (str): path to log directory
        level (str): loglevel default: INFO
        rotation: time, quantity to log rotation
        retention: time, quantity to log retention
    """
    # redirecting stdout/stderr to log file
    sys.stdout = StreamToLogger()
    sys.stderr = StreamToLogger()

    logger.remove()
    # show stdout, stderr on console, write log to log file
    # logger.add(sys.__stdout__, format=fmt, level=level.upper())
    logger.add(sys.__stderr__, level=level.upper())

    if rotation and retention is None:
        logger.add(log_file_path, level=level.upper(), rotation=rotation)
    elif retention and rotation is None:
        logger.add(log_file_path, level=level.upper(), retention=retention)
    elif rotation and retention:
        logger.add(
            log_file_path,
            level=level.upper(),
            rotation=rotation,
            retention=retention,
        )
    else:
        logger.add(log_file_path, level=level.upper())

    logger.level('SYSTEM', no=100, icon='🤖', color='<blue>')
    return logger
