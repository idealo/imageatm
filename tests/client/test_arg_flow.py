import pytest
import shutil
from pathlib import Path
from imageatm.client.client import Config
from imageatm.client.commands import pipeline, train, evaluate, cloud, dataprep


p = Path(__file__)
TEST_CONFIG_PIPE = p.resolve().parent / 'test_configs' / 'config_arg_flow_all.yml'
TEST_CONFIG_DATAPREP = p.resolve().parent / 'test_configs' / 'config_arg_flow_dataprep.yml'
TEST_CONFIG_TRAIN = p.resolve().parent / 'test_configs' / 'config_arg_flow_train.yml'
TEST_CONFIG_EVAL = p.resolve().parent / 'test_configs' / 'config_arg_flow_eval.yml'

TEST_SAMPLES = Path('tests/data/test_samples/test_arg_flow.json')
TEST_IMAGE_DIR = Path('tests/data/test_images')
TEST_IMAGE_DIR_RES = Path('tests/data/test_images_resized')
TEST_JOB_DIR = Path('tests/data/test_arg_flow')


@pytest.fixture(scope='session', autouse=True)
def tear_down(request):
    def remove_job_dir():
        shutil.rmtree(TEST_JOB_DIR)

    request.addfinalizer(remove_job_dir)


class TestArgFlow(object):
    def test_dataprep(self):
        config = Config()

        assert not TEST_IMAGE_DIR_RES.exists()
        assert not Path(TEST_JOB_DIR / 'class_mapping.json').exists()
        assert not Path(TEST_JOB_DIR / 'test_samples.json').exists()
        assert not Path(TEST_JOB_DIR / 'train_samples.json').exists()
        assert not Path(TEST_JOB_DIR / 'val_samples.json').exists()

        dataprep(config, config_file=TEST_CONFIG_DATAPREP)

        assert config.data_prep['run'] == True
        assert config.data_prep['job_dir'] == str(TEST_JOB_DIR)
        assert config.data_prep['samples_file'] == str(TEST_SAMPLES)
        assert config.data_prep['image_dir'] == str(TEST_IMAGE_DIR)
        assert config.data_prep['resize'] == True

        assert config.train['run'] == False
        assert config.evaluate['run'] == False
        assert config.cloud['run'] == False

        assert TEST_IMAGE_DIR_RES.exists()
        assert Path(TEST_JOB_DIR / 'class_mapping.json').exists()
        assert Path(TEST_JOB_DIR / 'test_samples.json').exists()
        assert Path(TEST_JOB_DIR / 'train_samples.json').exists()
        assert Path(TEST_JOB_DIR / 'val_samples.json').exists()

    def test_train(self):
        config = Config()

        assert not list(Path(TEST_JOB_DIR / 'models').glob('*.hdf5'))

        train(config, config_file=TEST_CONFIG_TRAIN)

        assert config.train['run'] == True
        assert config.train['cloud'] == False
        assert config.train['job_dir'] == str(TEST_JOB_DIR)
        assert config.train['image_dir'] == str(TEST_IMAGE_DIR_RES)

        assert config.data_prep['run'] == False
        assert config.evaluate['run'] == False
        assert config.cloud['run'] == False

        assert list(Path(TEST_JOB_DIR / 'models').glob('*.hdf5'))

    def test_evaluate(self):
        config = Config()

        assert not list(TEST_JOB_DIR.glob('*/confusion_matrix.pdf'))
        assert not list(TEST_JOB_DIR.glob('*/test_set_distribution.pdf'))

        evaluate(config, config_file=TEST_CONFIG_EVAL)

        assert config.data_prep['run'] == False
        assert config.train['run'] == False
        assert config.cloud['run'] == False

        assert config.evaluate['run'] == True
        assert config.evaluate['job_dir'] == str(TEST_JOB_DIR)
        assert config.evaluate['image_dir'] == str(TEST_IMAGE_DIR_RES)
        assert list(TEST_JOB_DIR.glob('*/confusion_matrix.pdf'))
        assert list(TEST_JOB_DIR.glob('*/test_set_distribution.pdf'))

    def test_pipeline(self):
        config = Config()

        pipeline(config, config_file=TEST_CONFIG_PIPE)

        assert config.data_prep['run'] == True
        assert config.data_prep['job_dir'] == str(TEST_JOB_DIR)
        assert config.data_prep['samples_file'] == str(TEST_SAMPLES)
        assert config.data_prep['image_dir'] == TEST_IMAGE_DIR_RES
        assert config.data_prep['resize'] == True

        assert config.train['run'] == True
        assert config.train['cloud'] == False

        assert config.evaluate['run'] == True

        assert config.cloud['run'] == False
        assert config.cloud['provider'] == 'aws'
        assert config.cloud['tf_dir'] == 'cloud/aws'
        assert config.cloud['region'] == 'eu-west-1'
        assert config.cloud['vpc_id'] == 'abc'
        assert config.cloud['instance_type'] == 't2.micro'
        assert config.cloud['bucket'] == 's3://test_bucket'
        assert config.cloud['destroy'] == True
        assert config.cloud['cloud_tag'] == 'test_user'

        assert list(TEST_JOB_DIR.glob('*/confusion_matrix.pdf'))
        assert list(TEST_JOB_DIR.glob('*/test_set_distribution.pdf'))
        assert list(Path(TEST_JOB_DIR / 'models').glob('*.hdf5'))
