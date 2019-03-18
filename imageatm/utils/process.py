import re
import sys
import tqdm
import subprocess
import logging
from typing import Callable
from multiprocessing import Pool, cpu_count


def parallelise(function: Callable, data: list) -> list:
    processes = cpu_count()
    pool = Pool(processes=processes)
    results = list(tqdm.tqdm(pool.imap(function, data), total=len(data)))
    pool.close()
    pool.join()
    return results


def run_cmd(cmd: str, logger: logging.Logger, level: str = 'debug', return_output: bool = False):
    # filter out ANSI color and font formatting
    ansi_re = re.compile(r'\x1b\[[0-9;]*m')

    p = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        universal_newlines=True,
        shell=True,
        bufsize=1,
    )

    # stream process stdout to logger if available
    stdout = []
    line = ''
    while True:
        inchar = p.stdout.read(1)
        line += inchar

        if line[-1:] == '\n':
            new_line = re.sub(ansi_re, '', line[:-1])

            if level == 'debug':
                logger.debug(new_line)
            else:
                logger.info(new_line)

            stdout.append(new_line)
            line = ''

        if not inchar:
            sys.stdout.flush()
            break

    output, error = p.communicate()

    if p.returncode != 0 and error:
        logger.error(error, exc_info=True)
        raise Exception('{}'.format(error))
    if return_output:
        return stdout[-1]
