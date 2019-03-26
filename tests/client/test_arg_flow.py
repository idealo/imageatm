import pytest
import shutil
import logging
from pathlib import Path
from imageatm.client.client import Config
from imageatm.client.commands import pipeline, train, evaluate, cloud
from imageatm.client.config import (
    config_set_job_dir,
    config_set_image_dir,
    update_component_configs,
    update_config,
    get_diff,
)

p = Path(__file__)
TEST_CONFIG_PIPE = p.resolve().parent / 'test_configs' / 'config_arg_flow_all.yml'
# TEST_CONFIG_DATA_PREP = p.resolve().parent / 'test_configs' / 'config_data_prep.yml'
# TEST_CONFIG_TRAIN = p.resolve().parent / 'test_configs' / 'config_train.yml'
# TEST_CONFIG_EVAL = p.resolve().parent / 'test_configs' / 'config_evaluate.yml'

TEST_SAMPLES = Path('../../../data/test_samples/test_str_labels.json')
TEST_IMAGE_DIR = Path('../../../data/test_images')
TEST_JOB_DIR = Path('../../../data/test_data_prep/job_dir')

'''
@pytest.fixture(scope='session', autouse=True)
def tear_down(request):
    def remove_job_dir():
        shutil.rmtree(Path('../../../data/test_evaluate/'))
        shutil.rmtree(Path('../../../data/test_train/'))
        shutil.rmtree(Path('../../../data/test_data_prep'))

    request.addfinalizer(remove_job_dir)
'''


class TestArgFlow(object):
    def test_pipeline(self, mocker):
        mp_dp_run = mocker.patch('imageatm.components.data_prep.DataPrep.run')
        mp_t_run = mocker.patch('imageatm.components.training.Training.run')
        mp_e_run = mocker.patch('imageatm.components.evaluation.Evaluation.run')
        mp_dp_load_json = mocker.patch('imageatm.components.data_prep.load_json', return_value={})
        mp_t_load_json = mocker.patch('imageatm.components.training.load_json', return_value={})
        mp_e_load_json = mocker.patch('imageatm.components.evaluation.load_json', return_value={})
        mp_load_best_model = mocker.patch(
            'imageatm.components.evaluation.Evaluation._load_best_model'
        )
        mp_create_evaluation_dir = mocker.patch(
            'imageatm.components.evaluation.Evaluation._create_evaluation_dir'
        )
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        config = Config()

        pipeline(config, config_file=TEST_CONFIG_PIPE)

        mp_dp_run.assert_called_with(resize=True)
        mp_dp_load_json.assert_called_with(TEST_SAMPLES)

        assert config.data_prep['run'] == True
        assert config.data_prep['job_dir'] == str(TEST_JOB_DIR)
        assert config.data_prep['samples_file'] == str(TEST_SAMPLES)
        assert config.data_prep['image_dir'] == TEST_IMAGE_DIR
        assert config.data_prep['resize'] == True

        assert config.train['run'] == True
        assert config.train['cloud'] == False

        assert config.evaluate['run'] == True

        assert config.cloud['run'] == True
        assert config.cloud['provider'] == 'aws'
        assert config.cloud['tf_dir'] == 'cloud/aws'
        assert config.cloud['region'] == 'eu-west-1'
        assert config.cloud['vpc_id'] == 'abc'
        assert config.cloud['instance_type'] == 't2.micro'
        assert config.cloud['bucket'] == 's3://test_bucket'
        assert config.cloud['destroy'] == True
        assert config.cloud['cloud_tag'] == 'test_user'
