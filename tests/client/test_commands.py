import pytest
import shutil
import logging
from pathlib import Path
from imageatm.client.client import Config
from imageatm.client.commands import pipeline, train, evaluate, cloud

p = Path(__file__)

TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_train.yml'

# TODO: Add relative path
TEST_JOB_DIR = 'test_data_prep'

@pytest.fixture(scope='session', autouse=True)
def tear_down(request):
    def remove_job_dir():
        shutil.rmtree(TEST_JOB_DIR)

    request.addfinalizer(remove_job_dir)

def mock_scripts(mocker):
    m_dp = mocker.patch('imageatm.scripts.run_data_prep')
    m_tc = mocker.patch('imageatm.scripts.run_training_cloud')
    m_t = mocker.patch('imageatm.scripts.run_training')
    m_e = mocker.patch('imageatm.scripts.run_evaluation')
    m_c = mocker.patch('imageatm.scripts.run_cloud')

    m_l = mocker.patch('imageatm.client.commands.get_logger')
    logger = logging.getLogger()
    m_l.return_value = logger

    return m_dp, m_tc, m_t, m_e, m_c, m_l


def test_pipeline(mocker):
    # assert that only data_prep gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_data_prep.yml'

    config = Config()

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    pipeline(config, config_file=TEST_CONFIG_FILE)

    m_dp.assert_called()
    m_tc.assert_not_called()
    m_t.assert_not_called()
    m_e.assert_not_called()
    m_c.assert_not_called()

    # assert that only train gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_train.yml'

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    pipeline(config, config_file=TEST_CONFIG_FILE)

    m_dp.assert_not_called()
    m_tc.assert_not_called()
    m_t.assert_called()
    m_e.assert_not_called()
    m_c.assert_not_called()

    # assert that only train cloud gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_train_cloud.yml'

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    pipeline(config, config_file=TEST_CONFIG_FILE)

    m_dp.assert_not_called()
    m_tc.assert_called()
    m_t.assert_not_called()
    m_e.assert_not_called()
    m_c.assert_not_called()

    # assert that only evaluate gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_evaluate.yml'

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    pipeline(config, config_file=TEST_CONFIG_FILE)

    m_dp.assert_not_called()
    m_tc.assert_not_called()
    m_t.assert_not_called()
    m_e.assert_called()
    m_c.assert_not_called()

    # assert that only cloud gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_cloud.yml'

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    pipeline(config, config_file=TEST_CONFIG_FILE)

    m_dp.assert_not_called()
    m_tc.assert_not_called()
    m_t.assert_not_called()
    m_e.assert_not_called()
    m_c.assert_called()

    # assert that all components get run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_all.yml'

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    pipeline(config, config_file=TEST_CONFIG_FILE)

    m_dp.assert_called()
    m_tc.assert_not_called()
    m_t.assert_called()
    m_e.assert_called()
    m_c.assert_called()


def test_train(mocker):
    # assert that train gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_train.yml'

    config = Config()

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    train(config, config_file=TEST_CONFIG_FILE)

    m_tc.assert_not_called()
    m_t.assert_called()

    # assert that train cloud gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_train_cloud.yml'

    config = Config()

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    train(config, config_file=TEST_CONFIG_FILE)

    m_tc.assert_called()
    m_t.assert_not_called()


def test_evaluate(mocker):
    # assert that evaluate gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_evaluate.yml'

    config = Config()

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    evaluate(config, config_file=TEST_CONFIG_FILE)

    m_e.assert_called()

    # assert that train gets not run
    # even though run=False in config, if user calls evaluate command we want it to run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_evaluate.yml'

    config = Config()

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    evaluate(config, config_file=TEST_CONFIG_FILE)

    m_e.assert_called()


def test_cloud(mocker):
    # assert that evaluate gets run
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_cloud.yml'

    config = Config()

    m_dp, m_tc, m_t, m_e, m_c, m_l = mock_scripts(mocker)

    cloud(config, config_file=TEST_CONFIG_FILE)

    m_c.assert_called()
