import logging
import pathlib


FMT_MSG = "[%(asctime)s | %(name)s] %(message)s"
FMT_FULL = "[%(asctime)s | %(name)s | %(filename)s L%(lineno)s (%(funcName)s)] %(message)s"
FMT_TIME = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    reset = "\x1b[0m"
    red = "\x1b[31m"
    bold_red = "\x1b[31;1m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"

    FORMATS = {
        logging.DEBUG: f"{blue}{FMT_FULL}{reset}",
        logging.INFO: f"{green}{FMT_MSG}{reset}",
        logging.WARNING: f"{yellow}{FMT_FULL}{reset}",
        logging.ERROR: f"{red}{FMT_FULL}{reset}",
        logging.CRITICAL: f"{bold_red}{FMT_FULL}{reset}"
    }

    def format(self, record):
        formatter = logging.Formatter(self.FORMATS.get(record.levelno), datefmt=FMT_TIME)
        return formatter.format(record)


class NormalFomatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: FMT_FULL,
        logging.INFO: FMT_MSG,
        logging.WARNING: FMT_FULL,
        logging.ERROR: FMT_FULL,
        logging.CRITICAL: FMT_FULL
    }

    def format(self, record):
        formatter = logging.Formatter(self.FORMATS.get(record.levelno), datefmt=FMT_TIME)
        return formatter.format(record)


def init_logger(
        name: str,
        level: int = logging.INFO,
        out_path = None,
        colored: bool = True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if colored:
        formatter = ColoredFormatter()
    else:
        formatter = NormalFomatter()
    
    stream_handler = logging.StreamHandler()
    # stream_handler.setLevel(level) # Not use this to follow logger's level
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if out_path is not None:
        opath = pathlib.Path(out_path)
        if not opath.parent.exists():
            opath.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(filename=opath)
        # file_handler.setLevel(level) # Not use this to follow logger's level
        file_handler.setFormatter(NormalFomatter()) # Use normal formatter in file stream
        logger.addHandler(file_handler)
    
    return logger