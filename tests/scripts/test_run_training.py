import pytest
import shutil
from pathlib import Path
from imageatm.scripts import run_training
from imageatm.components.training import Training


TEST_IMAGE_DIR = Path('./tests/data/test_images').resolve()
TEST_JOB_DIR = Path('./tests/data/test_train_job').resolve()
TEST_BATCH_SIZE = 64
TEST_DROPOUT_RATE = 0.75
TEST_BASE_MODEL_NAME = 'MobileNet'
TEST_LOSS = 'test_loss'
TEST_EPOCHS_TRAIN_DENSE = 2
TEST_EPOCHS_TRAIN_ALL = 10
TEST_LEARNING_RATE_DENSE = 0.001
TEST_LEARNING_RATE_ALL = 0.0001


class TestRunEvaluation(object):
    def test_run_training_1(self, mocker):
        mp_run = mocker.patch('imageatm.components.training.Training.run')
        mocker.patch('imageatm.components.training.Training.__init__')
        Training.__init__.return_value = None

        run_training(image_dir=TEST_IMAGE_DIR, job_dir=TEST_JOB_DIR)
        mp_run.assert_called_once()
        Training.__init__.assert_called_with(image_dir=TEST_IMAGE_DIR, job_dir=TEST_JOB_DIR)

    def test_run_training_2(self, mocker):
        mp_run = mocker.patch('imageatm.components.training.Training.run')
        mocker.patch('imageatm.components.training.Training.__init__')
        Training.__init__.return_value = None
        run_training(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            epochs_train_dense=TEST_EPOCHS_TRAIN_DENSE,
            epochs_train_all=TEST_EPOCHS_TRAIN_ALL,
            learning_rate_dense=TEST_LEARNING_RATE_DENSE,
            learning_rate_all=TEST_LEARNING_RATE_ALL,
            batch_size=TEST_BATCH_SIZE,
            dropout_rate=TEST_DROPOUT_RATE,
            base_model_name=TEST_BASE_MODEL_NAME,
            loss=TEST_LOSS,
        )
        mp_run.assert_called_once()
        Training.__init__.assert_called_with(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            epochs_train_dense=TEST_EPOCHS_TRAIN_DENSE,
            epochs_train_all=TEST_EPOCHS_TRAIN_ALL,
            learning_rate_dense=TEST_LEARNING_RATE_DENSE,
            learning_rate_all=TEST_LEARNING_RATE_ALL,
            batch_size=TEST_BATCH_SIZE,
            dropout_rate=TEST_DROPOUT_RATE,
            base_model_name=TEST_BASE_MODEL_NAME,
            loss=TEST_LOSS,
        )
