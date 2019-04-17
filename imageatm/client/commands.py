from typing import Optional
from pathlib import Path
from imageatm.utils.logger import get_logger
from imageatm.client.config import Config, validate_config, update_config, update_component_configs


def pipeline(
    config: Config,
    config_file: Path,
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
    resize: Optional[bool] = None,
    batch_size: Optional[int] = None,
    learning_rate_dense: Optional[float] = None,
    learning_rate_all: Optional[float] = None,
    epochs_train_dense: Optional[int] = None,
    epochs_train_all: Optional[int] = None,
    base_model_name: Optional[str] = None,
    cloud_tag: Optional[str] = None,
):
    """Runs the entire pipeline based on config file."""
    config = update_config(
        config=config,
        config_file=config_file,
        job_dir=job_dir,
        image_dir=image_dir,
        samples_file=samples_file,
        provider=provider,
        instance_type=instance_type,
        region=region,
        vpc_id=vpc_id,
        bucket=bucket,
        tf_dir=tf_dir,
        train_cloud=train_cloud,
        destroy=destroy,
        resize=resize,
        batch_size=batch_size,
        learning_rate_dense=learning_rate_dense,
        learning_rate_all=learning_rate_all,
        epochs_train_dense=epochs_train_dense,
        epochs_train_all=epochs_train_all,
        base_model_name=base_model_name,
        cloud_tag=cloud_tag,
    )

    validate_config(config, config.pipeline)

    Path(config.job_dir).resolve().mkdir(parents=True, exist_ok=True)

    logger = get_logger(__name__, config.job_dir)  # type: ignore

    if 'dataprep' in config.pipeline:
        from imageatm.scripts import run_dataprep

        logger.info(
            '\n********************************\n'
            '******* Data preparation *******\n'
            '********************************'
        )

        dp = run_dataprep(**config.dataprep)

        # update image_dir if images were resized
        if config.dataprep.get('resize', False):
            config.image_dir = dp.image_dir  # type: ignore
            config = update_component_configs(config)

    if 'train' in config.pipeline:
        logger.info(
            '\n********************************\n'
            '*********** Training ***********\n'
            '********************************'
        )

        if config.train.get('cloud'):
            from imageatm.scripts import run_training_cloud

            run_training_cloud(**{**config.cloud, **config.train})
        else:
            from imageatm.scripts import run_training

            run_training(**config.train)

    if 'evaluate' in config.pipeline:
        from imageatm.scripts import run_evaluation

        logger.info(
            '\n********************************\n'
            '********** Evaluation **********\n'
            '********************************'
        )

        run_evaluation(**config.evaluate)

    if 'cloud' in config.pipeline:
        from imageatm.scripts import run_cloud

        run_cloud(**config.cloud)


def dataprep(
    config: Config,
    config_file: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    samples_file: Optional[Path] = None,
    job_dir: Optional[Path] = None,
    resize: Optional[bool] = None,
):
    config = update_config(
        config=config,
        config_file=config_file,
        job_dir=job_dir,
        image_dir=image_dir,
        samples_file=samples_file,
        resize=resize,
    )

    config.dataprep['run'] = True
    validate_config(config, ['dataprep'])

    from imageatm.scripts import run_dataprep

    run_dataprep(**config.dataprep)


def train(
    config,
    config_file: Optional[Path] = None,
    job_dir: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    provider: Optional[str] = None,
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    vpc_id: Optional[str] = None,
    bucket: Optional[str] = None,
    tf_dir: Optional[Path] = None,
    train_cloud: Optional[bool] = None,
    destroy: Optional[bool] = None,
    batch_size: Optional[int] = None,
    learning_rate_dense: Optional[float] = None,
    learning_rate_all: Optional[float] = None,
    epochs_train_dense: Optional[int] = None,
    epochs_train_all: Optional[int] = None,
    base_model_name: Optional[str] = None,
    cloud_tag: Optional[str] = None,
):
    config = update_config(
        config=config,
        config_file=config_file,
        job_dir=job_dir,
        image_dir=image_dir,
        provider=provider,
        instance_type=instance_type,
        region=region,
        vpc_id=vpc_id,
        bucket=bucket,
        tf_dir=tf_dir,
        train_cloud=train_cloud,
        destroy=destroy,
        batch_size=batch_size,
        learning_rate_dense=learning_rate_dense,
        learning_rate_all=learning_rate_all,
        epochs_train_dense=epochs_train_dense,
        epochs_train_all=epochs_train_all,
        base_model_name=base_model_name,
        cloud_tag=cloud_tag,
    )

    config.train['run'] = True

    validate_config(config, ['train'])

    if config.train.get('cloud'):
        from imageatm.scripts import run_training_cloud

        run_training_cloud(**{**config.cloud, **config.train})
    else:
        from imageatm.scripts import run_training

        run_training(**config.train)


def evaluate(
    config: Config,
    config_file: Optional[Path] = None,
    image_dir: Optional[Path] = None,
    job_dir: Optional[Path] = None,
):
    config = update_config(
        config=config, config_file=config_file, job_dir=job_dir, image_dir=image_dir
    )

    config.evaluate['run'] = True
    validate_config(config, ['evaluate'])

    from imageatm.scripts import run_evaluation

    run_evaluation(**config.evaluate)


def cloud(
    config,
    job_dir: Optional[Path] = None,
    config_file: Optional[Path] = None,
    provider: Optional[str] = None,
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    vpc_id: Optional[str] = None,
    bucket: Optional[str] = None,
    tf_dir: Optional[Path] = None,
    train_cloud: Optional[bool] = None,
    destroy: Optional[bool] = None,
    no_destroy: Optional[bool] = None,
    cloud_tag: Optional[str] = None,
):
    config = update_config(
        config=config,
        job_dir=job_dir,
        config_file=config_file,
        provider=provider,
        instance_type=instance_type,
        region=region,
        vpc_id=vpc_id,
        bucket=bucket,
        tf_dir=tf_dir,
        train_cloud=train_cloud,
        destroy=destroy,
        no_destroy=no_destroy,
        cloud_tag=cloud_tag,
    )

    config.cloud['run'] = True
    validate_config(config, ['cloud'])

    from imageatm.scripts import run_cloud

    run_cloud(**config.cloud)
