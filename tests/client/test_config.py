from pathlib import Path
from imageatm.client.client import Config
from imageatm.client.config import update_component_configs, update_config, get_diff

p = Path(__file__)


def test_update_component_configs():
    config = Config()
    config.image_dir = 'test_image'
    config.job_dir = 'test_job'

    result = update_component_configs(config)

    assert result.dataprep['image_dir'] == 'test_image'
    assert result.train['image_dir'] == 'test_image'
    assert result.evaluate['image_dir'] == 'test_image'

    assert result.dataprep['job_dir'] == 'test_job'
    assert result.train['job_dir'] == 'test_job'
    assert result.evaluate['job_dir'] == 'test_job'


def test_update_config():
    # check that defaults are being set
    config = Config()

    result = update_config(config)

    assert result.train == {'cloud': False}
    assert result.dataprep == {'resize': False}
    assert result.cloud == {}
    assert result.evaluate == {'report_html': False, 'report_pdf': False}

    # check that defaults, image_dir, and job_dir are being set
    config = Config()
    config.image_dir = 'test_image'
    config.job_dir = 'test_job'

    result = update_config(config)

    assert result.train == {'cloud': False, 'image_dir': 'test_image', 'job_dir': 'test_job'}
    assert result.dataprep == {'resize': False, 'image_dir': 'test_image', 'job_dir': 'test_job'}
    assert result.cloud == {'job_dir': 'test_job'}
    assert result.evaluate == {'image_dir': 'test_image', 'job_dir': 'test_job', 'report_html': False, 'report_pdf': False}

    # check that config file gets populated correctly
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_train.yml'

    config = Config()

    result = update_config(config, config_file=TEST_CONFIG_FILE)

    assert result.train == {
        'run': True,
        'cloud': False,
        'image_dir': 'test_train/images',
        'job_dir': 'test_train/job_dir',
    }
    assert result.dataprep == {
        'run': False,
        'resize': True,
        'image_dir': 'test_train/images',
        'job_dir': 'test_train/job_dir',
        'samples_file': 'test_dataprep/samples.json',
    }
    assert result.cloud == {
        'run': False,
        'provider': 'aws',  # supported providers ['aws']
        'tf_dir': 'cloud/aws',
        'region': 'eu-west-1',  # supported regions ['eu-west-1', 'eu-central-1']
        'vpc_id': 'abc',
        'instance_type': 't2.micro',  # supported instances ['p2.xlarge']
        'bucket': 's3://test_bucket',  # s3 bucket needs to exist, will not be created/destroyed by terraform
        'destroy': True,
        'job_dir': 'test_train/job_dir',
        'cloud_tag': 'test_user',
    }
    assert result.evaluate == {
        'run': False,
        'image_dir': 'test_train/images',
        'job_dir': 'test_train/job_dir',
        'report_pdf': False,
        'report_html': False,
    }

    # test that options overwrite config file
    TEST_CONFIG_FILE = p.resolve().parent / 'test_configs' / 'config_train.yml'

    config = Config()

    result = update_config(
        config,
        config_file=TEST_CONFIG_FILE,
        image_dir='test_image',
        job_dir='test_job',
        region='eu-central-1',
    )

    assert result.train == {
        'run': True,
        'cloud': False,
        'image_dir': 'test_image',
        'job_dir': 'test_job',
    }

    assert result.dataprep == {
        'run': False,
        'resize': True,
        'image_dir': 'test_image',
        'job_dir': 'test_job',
        'samples_file': 'test_dataprep/samples.json',
    }

    assert result.cloud == {
        'run': False,
        'provider': 'aws',
        'tf_dir': 'cloud/aws',
        'region': 'eu-central-1',
        'vpc_id': 'abc',
        'instance_type': 't2.micro',
        'bucket': 's3://test_bucket',
        'destroy': True,
        'bucket': 's3://test_bucket',
        'job_dir': 'test_job',
        'cloud_tag': 'test_user',
    }

    assert result.evaluate == {
        'run': False,
        'image_dir': 'test_image',
        'job_dir': 'test_job',
         'report_pdf': False,
         'report_html': False,
    }


def test_get_diff():
    # test required fields missing
    required_keys = ['a', 'b']
    optional_keys = ['c', 'd']

    config = {'a': 124}

    expected = ['train config: missing required parameters [b]\n']
    result = get_diff('train', config, required_keys, optional_keys)

    assert result == expected

    # test no required fields missing
    required_keys = ['a', 'b']
    optional_keys = ['c', 'd']

    config = {'a': 124, 'b': 234}

    expected = []
    result = get_diff('train', config, required_keys, optional_keys)

    assert result == expected

    # test all keys allowed
    required_keys = ['a', 'b']
    optional_keys = ['c', 'd']

    config = {'a': 124, 'b': 234, 'c': 456, 'd': 678}

    expected = []
    result = get_diff('train', config, required_keys, optional_keys)

    assert result == expected

    # test not all keys allowed
    required_keys = ['a', 'b']
    optional_keys = ['c', 'd']

    config = {'a': 124, 'b': 234, 'c': 456, 'e': 678}

    expected = ['train config: [e] not in allowed parameters [a, b, c, d]\n']
    result = get_diff('train', config, required_keys, optional_keys)

    assert result == expected

    # test both allowed and rrequired keys
    required_keys = ['a', 'b']
    optional_keys = ['c', 'd']

    config = {'a': 124, 'c': 456, 'e': 678}

    expected = [
        'train config: missing required parameters [b]\n',
        'train config: [e] not in allowed parameters [a, b, c, d]\n',
    ]
    result = get_diff('train', config, required_keys, optional_keys)

    assert result == expected
