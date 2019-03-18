import pytest
import shutil
from pathlib import Path
from imageatm.scripts import run_evaluation
from imageatm.components.evaluation import Evaluation


TEST_IMAGE_DIR = Path('./tests/data/test_images').resolve()
TEST_JOB_DIR = Path('./tests/data/test_train_job').resolve()
TEST_BATCH_SIZE = 16
TEST_BASE_MODEL_NAME = 'MobileNet'


class TestRunEvaluation(object):
    def test_run_evaluation(self, mocker):
        mocker.patch('imageatm.components.evaluation.load_model', return_value={})
        mp_run = mocker.patch('imageatm.components.evaluation.Evaluation.run')
        mocker.patch('imageatm.components.evaluation.Evaluation.__init__')
        Evaluation.__init__.return_value = None

        run_evaluation(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            batch_size=TEST_BATCH_SIZE,
            base_model_name=TEST_BASE_MODEL_NAME,
        )
        mp_run.assert_called_once()
        Evaluation.__init__.assert_called_with(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            batch_size=TEST_BATCH_SIZE,
            base_model_name=TEST_BASE_MODEL_NAME,
        )

        run_evaluation(image_dir=TEST_IMAGE_DIR, job_dir=TEST_JOB_DIR)
        Evaluation.__init__.assert_called_with(image_dir=TEST_IMAGE_DIR, job_dir=TEST_JOB_DIR)
