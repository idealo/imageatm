import sys
import logging
from pathlib import Path
from typing import Union


def get_logger(name: str, job_dir: Union[Path, str]) -> logging.Logger:
    if isinstance(job_dir, str):
        job_dir = Path(job_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # stream handler ensures that logging events are passed to stdout
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter('%(message)s')
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

        # file handler ensures that logging events are passed to log file
        fh = logging.FileHandler(filename=job_dir / 'logs')
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            '%(asctime)s - %(module)s - %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    return logger
