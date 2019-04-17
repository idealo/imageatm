from pathlib import Path
from typing import Optional, List
from imageatm.utils.io import load_yaml


class Config:
    def __init__(self) -> None:
        # components
        self.dataprep: dict = {}
        self.train: dict = {}
        self.cloud: dict = {}
        self.evaluate: dict = {}

        # central parameters
        self.image_dir: Optional[Path] = None
        self.job_dir: Optional[Path] = None

        self.pipeline: list = []


def update_component_configs(config: Config) -> Config:
    """Populate central parameters to component configs."""
    if config.image_dir:
        config.dataprep['image_dir'] = config.image_dir
        config.train['image_dir'] = config.image_dir
        config.evaluate['image_dir'] = config.image_dir

    if config.job_dir:
        config.dataprep['job_dir'] = config.job_dir
        config.train['job_dir'] = config.job_dir
        config.evaluate['job_dir'] = config.job_dir
        config.cloud['job_dir'] = config.job_dir

    return config


def update_config(
    config: Config,
    config_file: Optional[Path] = None,
    job_dir: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    samples_file: Optional[Path] = None,
    provider: Optional[str] = None,
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    vpc_id: Optional[str] = None,
    bucket: Optional[str] = None,
    tf_dir: Optional[Path] = None,
    train_cloud: Optional[bool] = None,
    destroy: Optional[bool] = None,
    no_destroy: Optional[bool] = None,
    resize: Optional[bool] = None,
    batch_size: Optional[int] = None,
    learning_rate_dense: Optional[float] = None,
    learning_rate_all: Optional[float] = None,
    epochs_train_dense: Optional[int] = None,
    epochs_train_all: Optional[int] = None,
    base_model_name: Optional[str] = None,
    cloud_tag: Optional[str] = None,
) -> Config:

    # set defaults
    config.train['cloud'] = False
    config.dataprep['resize'] = False
    config.pipeline = []

    # load central config file
    if config_file:
        config_yml: dict = load_yaml(config_file)  # type: ignore

        # populate parameters from config file
        # parameters from config file overwrite defaults
        config.image_dir = config_yml.get('image_dir')
        config.job_dir = config_yml.get('job_dir')

        config.dataprep = {**config.dataprep, **config_yml.get('dataprep', {})}
        config.train = {**config.train, **config_yml.get('train', {})}
        config.evaluate = {**config.evaluate, **config_yml.get('evaluate', {})}
        config.cloud = {**config.cloud, **config_yml.get('cloud', {})}

        # set pipeline
        components = ['dataprep', 'train', 'evaluate', 'cloud']
        config.pipeline = [i for i in components if config_yml.get(i, {}).get('run')]

    # set options
    if job_dir is not None:
        config.job_dir = job_dir

    if image_dir is not None:
        config.image_dir = image_dir

    if samples_file is not None:
        config.dataprep['samples_file'] = samples_file

    if provider is not None:
        config.cloud['provider'] = provider

    if instance_type is not None:
        config.cloud['instance_type'] = instance_type

    if region is not None:
        config.cloud['region'] = region

    if vpc_id is not None:
        config.cloud['vpc_id'] = vpc_id

    if bucket is not None:
        config.cloud['bucket'] = bucket

    if tf_dir is not None:
        config.cloud['tf_dir'] = tf_dir

    if train_cloud is True:
        config.train['cloud'] = True

    if destroy is True:
        config.cloud['destroy'] = True

    if no_destroy is True:
        config.cloud['destroy'] = False

    if resize is True:
        config.dataprep['resize'] = True

    if batch_size is not None:
        config.train['batch_size'] = batch_size

    if learning_rate_dense is not None:
        config.train['learning_rate_dense'] = learning_rate_dense

    if learning_rate_all is not None:
        config.train['learning_rate_all'] = learning_rate_all

    if epochs_train_dense is not None:
        config.train['epochs_train_dense'] = epochs_train_dense

    if epochs_train_all is not None:
        config.train['epochs_train_all'] = epochs_train_all

    if base_model_name is not None:
        config.train['base_model_name'] = base_model_name

    if cloud_tag is not None:
        config.cloud['cloud_tag'] = cloud_tag

    config = update_component_configs(config)

    return config


def get_diff(
    name: str, config: dict, required_keys: List[str], optional_keys: List[str]
) -> List[str]:
    allowed_keys = required_keys + optional_keys

    msg = []
    # check that all required keys are in config keys
    diff = list(set(required_keys).difference(config.keys()))
    if diff:
        msg.append('{} config: missing required parameters [{}]\n'.format(name, ', '.join(diff)))

    # check that config keys are in allowed keys
    diff = list(set(config.keys()).difference(allowed_keys))
    if diff:
        msg.append(
            '{} config: [{}] not in allowed parameters [{}]\n'.format(
                name, ', '.join(diff), ', '.join(allowed_keys)
            )
        )

    return msg


def val_dataprep(config: dict) -> List[str]:
    required_keys = ['image_dir', 'job_dir', 'samples_file', 'run']
    optional_keys = ['resize']

    return get_diff('dataprep', config, required_keys, optional_keys)


def val_train(config: dict) -> List[str]:
    required_keys = ['image_dir', 'job_dir', 'cloud', 'run']
    optional_keys = [
        'batch_size',
        'learning_rate_dense',
        'learning_rate_all',
        'epochs_train_dense',
        'epochs_train_all',
        'base_model_name',
    ]

    return get_diff('train', config, required_keys, optional_keys)


def val_evaluate(config: dict) -> List[str]:
    required_keys = ['image_dir', 'job_dir', 'run']
    optional_keys: list = []

    return get_diff('evaluate', config, required_keys, optional_keys)


def val_cloud(config: dict) -> List[str]:
    allowed_providers = ['aws']

    assert config.get('provider') is not None, 'Config error: cloud config: missing provider'

    provider = config.get('provider')
    assert (
        provider in allowed_providers
    ), 'Config error: cloud config: {} not in allowed providers [{}]'.format(
        provider, *allowed_providers
    )

    required_keys = {
        'aws': [
            'run',
            'job_dir',
            'tf_dir',
            'region',
            'vpc_id',
            'instance_type',
            'bucket',
            'destroy',
            'provider',
            'cloud_tag',
        ]
    }
    optional_keys: list = []

    return get_diff('cloud', config, required_keys[provider], optional_keys)


def validate_config(config: Config, components: Optional[list]):
    msgs: list = []

    if components:
        for component in components:
            config_component = getattr(config, component)

            if component == 'dataprep':
                msgs += val_dataprep(config_component)

            if component == 'train':
                msgs += val_train(config_component)

            if component == 'evaluate':
                msgs += val_evaluate(config_component)

            if component == 'cloud':
                msgs += val_cloud(config_component)

    assert len(msgs) == 0, '\nConfig error:\n{}'.format(''.join(msgs))
