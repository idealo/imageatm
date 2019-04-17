from pathlib import Path
from imageatm.scripts import run_dataprep
from imageatm.components.dataprep import DataPrep

TEST_IMAGE_DIR = Path('./tests/data/test_images').resolve()
TEST_JOB_DIR = Path('./tests/data/test_train_job').resolve()
TEST_STR_FILE = Path('./tests/data/test_samples/test_str_labels.json').resolve()
TEST_BATCH_SIZE = 16
TEST_BASE_MODEL_NAME = 'MobileNet'


class TestRunDataPrep(object):
    def test_run_dataprep(self, mocker):
        mp_run = mocker.patch('imageatm.components.dataprep.DataPrep.run')
        mocker.patch('imageatm.components.dataprep.DataPrep.__init__')
        DataPrep.__init__.return_value = None

        run_dataprep(
            image_dir=TEST_IMAGE_DIR, job_dir=TEST_JOB_DIR, samples_file=TEST_STR_FILE, resize=True
        )
        mp_run.assert_called_with(resize=True)
        DataPrep.__init__.assert_called_with(
            image_dir=TEST_IMAGE_DIR, job_dir=TEST_JOB_DIR, samples_file=TEST_STR_FILE
        )

        run_dataprep(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            samples_file=TEST_STR_FILE,
            resize=True,
            min_class_size=1,
            test_size=1,
            val_size=1,
            part_size=1,
        )
        DataPrep.__init__.assert_called_with(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            samples_file=TEST_STR_FILE,
            min_class_size=1,
            test_size=1,
            val_size=1,
            part_size=1,
        )

        run_dataprep(image_dir=TEST_IMAGE_DIR, job_dir=TEST_JOB_DIR, samples_file=TEST_STR_FILE)
        mp_run.assert_called_with(resize=False)
