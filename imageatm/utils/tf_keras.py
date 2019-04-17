import numpy as np
from six import iteritems
from multiprocessing import cpu_count
from typing import Tuple, List
from logging import Logger
from tensorflow.python.client import device_lib
from keras.models import load_model as load_model_keras
from keras.callbacks import Callback
from keras.engine.training import Model
from pathlib import Path


def load_model(model_path: Path) -> Model:
    return load_model_keras(str(model_path))


def use_multiprocessing() -> Tuple[bool, int]:
    if _get_available_gpus():
        # if GPU is available, use all available CPUs for batch preprocessing
        use_multiprocessing = True
        num_workers = cpu_count()
    else:
        # device = 'CPU'
        use_multiprocessing = False
        num_workers = 1
    return use_multiprocessing, num_workers


def _get_available_gpus() -> List[str]:
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


class LoggingMetrics(Callback):
    """Callback for logging metrics at the end of each epoch.

    Args:
        logger: Root logger.
    """

    def __init__(self, logger: Logger) -> None:
        Callback.__init__(self)
        self.logger = logger
        self.format_epoch = 'Epoch: {} - {}'
        self.format_keyvalue = '{}: {:0.4f}'
        self.format_separator = ' - '

    def on_epoch_end(self, epoch: int, logs: dict = {}):
        values = self.format_separator.join(
            self.format_keyvalue.format(k, v) for k, v in iteritems(logs)
        )
        msg = self.format_epoch.format(epoch + 1, values)
        self.logger.debug(msg)


class LoggingModels(Callback):
    def __init__(
        self,
        filepath: Path,
        logger: Logger,
        monitor: str = 'val_loss',
        verbose: int = 0,
        save_best_only: bool = False,
        save_weights_only: bool = False,
        mode: str = 'auto',
        period: int = 1,
    ) -> None:
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.logger = logger

        if mode not in ['auto', 'min', 'max']:
            self.logger.warning(
                'ModelCheckpoint mode {} is unknown, fallback to auto mode.'.format(mode)
            )
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch: int, logs: dict = None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = Path(str(self.filepath).format(epoch=epoch + 1, **logs))
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    self.logger.warning(
                        'Can save best model only with {} available, skipping.'.format(self.monitor)
                    )
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            self.logger.info(
                                '\nEpoch {:05d} {} improved from {:0.5f} to {:0.5f},\nsaving model to {}'.format(
                                    epoch + 1, self.monitor, self.best, current, filepath
                                )
                            )
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(str(filepath), overwrite=True)
                        else:
                            self.model.save(str(filepath), overwrite=True)
                    else:
                        if self.verbose > 0:
                            self.logger.info(
                                '\nEpoch {:05d} {} did not improve from {:0.5f}'.format(
                                    epoch + 1, self.monitor, self.best
                                )
                            )
            else:
                if self.verbose > 0:
                    self.logger.debug(
                        '\nEpoch {:05d} saving model to {}'.format(epoch + 1, filepath)
                    )
                if self.save_weights_only:
                    self.model.save_weights(str(filepath), overwrite=True)
                else:
                    self.model.save(str(filepath), overwrite=True)
