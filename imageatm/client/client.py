import click
from imageatm.client import commands
from imageatm.client.config import Config


# creates config object that will be passed from cli to subcommands
pass_config = click.make_pass_decorator(Config, ensure=True)


# cli is the group parent, it gets run before any of the subcommands are run
@click.group()
@pass_config
def cli(config: Config):
    pass


@cli.command()
@click.argument('config-file', type=click.Path())
@click.option('--image-dir', type=click.Path(), help='Directory with image files.')
@click.option('--samples-file', type=click.Path(), help='JSON file with samples.')
@click.option(
    '--job-dir',
    type=click.Path(),
    help=('Directory with train, val, and test samples files and class_mapping file.'),
)
@click.option('--provider', help='Cloud provider, currently supported: [aws].')
@click.option('--instance-type', help='Cloud instance_type [aws].')
@click.option('--region', help='Cloud region [aws].')
@click.option('--vpc-id', help='Cloud VPC id [aws].')
@click.option('--bucket', help='Cloud bucket used for persistence [aws].')
@click.option('--tf-dir', help='Directory with Terraform configs [aws].')
@click.option('--train-cloud', is_flag=True, required=False, help='Run training in cloud [aws].')
@click.option('--destroy', is_flag=True, required=False, help='Destroys cloud.')
@click.option('--resize', is_flag=True, required=False, help='Resizes images in dataprep.')
@click.option('--batch-size', type=int, help='Batch size.', required=False)
@click.option(
    '--epochs-train-dense',
    type=int,
    help='Number of epochs train only dense layer.',
    required=False,
)
@click.option(
    '--epochs-train-all', type=int, help='Number of epochs train all layers.', required=False
)
@click.option(
    '--learning-rate-dense', type=float, help='Learning rate dense layers.', required=False
)
@click.option('--learning-rate-all', type=float, help='Learning rate all layers.', required=False)
@click.option(
    '--base-model-name', help='Pretrained CNN to be used for transfer learning.', required=False
)
@click.option(
    '--cloud-tag', help='Tag under which all cloud resources are created.', required=False
)
@pass_config
def pipeline(config: Config, **kwargs):
    """Runs all components for which run=True in config file.

    All activated (run=True) components from config file will be run in sequence. Options overwrite the config file.
    The config file is the only way to define pipeline components.

    Args:
        config-file: Central configuration file.
    """
    commands.pipeline(config, **kwargs)


@cli.command()
@click.option('--config-file', type=click.Path(), help='Central configuration file.')
@click.option('--image-dir', type=click.Path(), help='Directory with image files.')
@click.option('--samples-file', type=click.Path(), help='JSON file with samples.')
@click.option(
    '--job-dir',
    type=click.Path(),
    help=('Directory with train, val, and test samples files and class_mapping file.'),
)
@click.option(
    '--resize',
    is_flag=True,
    required=False,
    help='Resizes images and stores them in _resized subfolder.',
)
@pass_config
def dataprep(config: Config, **kwargs):
    """Run data preparation and create job dir.

    Creates a directory (job_dir) with the following files:

        - train_samples.json

        - val_samples.json

        - test_samples.json

        - class_mapping.json
    """
    commands.dataprep(config, **kwargs)


@cli.command()
@click.option('--config-file', type=click.Path(), help='Central configuration file.')
@click.option('--image-dir', type=click.Path(), help='Directory with image files.')
@click.option(
    '--job-dir',
    type=click.Path(),
    help=('Directory with train, val, and test samples files and class_mapping file.'),
)
@click.option('--provider', help='Cloud provider, currently supported: [aws].')
@click.option('--instance-type', help='Cloud instance_type [aws].')
@click.option('--region', help='Cloud region [aws].')
@click.option('--vpc-id', help='Cloud VPC id [aws].')
@click.option('--bucket', help='Cloud bucket used for persistence [aws].')
@click.option('--tf-dir', help='Directory with Terraform configs [aws].')
@click.option('--train-cloud', is_flag=True, required=False, help='Run training in cloud [aws].')
@click.option('--destroy', is_flag=True, required=False, help='Destroys cloud.')
@click.option('--batch-size', type=int, help='Batch size.', required=False)
@click.option(
    '--epochs-train-dense',
    type=int,
    help='Number of epochs train only dense layer.',
    required=False,
)
@click.option(
    '--epochs-train-all', type=int, help='Number of epochs train all layers.', required=False
)
@click.option(
    '--learning-rate-dense', type=float, help='Learning rate dense layers.', required=False
)
@click.option('--learning-rate-all', type=float, help='Learning rate all layers.', required=False)
@click.option(
    '--base-model-name', help='Pretrained CNN to be used for transfer learning.', required=False
)
@click.option(
    '--cloud-tag', help='Tag under which all cloud resources are created.', required=False
)
@pass_config
def train(config: Config, **kwargs):
    """Train a CNN.

    Fine-tunes an ImageNet pre-trained CNN. The number of classes are derived from train_samples.json.
    After each epoch the model will be evaluated on val_samples.json.

    The best model (based on valuation accuracy) will be saved.

    Args:
        image_dir: Directory with image files.
        job_dir: Directory with train_samples, val_samples, and class_mapping.json.

    """
    commands.train(config, **kwargs)


@cli.command()
@click.option('--config-file', type=click.Path(), help='Central configuration file.')
@click.option('--image-dir', type=click.Path(), help='Directory with image files.')
@click.option(
    '--job-dir', type=click.Path(), help=('Directory with test samples files and trained model.')
)
@pass_config
def evaluate(config: Config, **kwargs):
    """Evaluate a trained model.

    Evaluation will be performed on test_samples.json.

    Args:
        image_dir: Directory with image files.
        job_dir: Directory with test_samples.json and class_mapping.json.
    """
    commands.evaluate(config, **kwargs)


@cli.command()
@click.option('--config-file', type=click.Path(), help='Central configuration file.')
@click.option(
    '--job-dir', type=click.Path(), help=('Directory with test samples files and trained model.')
)
@click.option('--provider', help='Cloud provider, currently supported: [aws].')
@click.option('--instance-type', help='Cloud instance_type [aws].')
@click.option('--region', help='Cloud region [aws].')
@click.option('--vpc-id', help='Cloud VPC id [aws].')
@click.option('--bucket', help='Cloud bucket used for persistence [aws].')
@click.option('--tf-dir', help='Directory with Terraform configs [aws].')
@click.option('--train-cloud', is_flag=True, required=False, help='Run training in cloud [aws].')
@click.option('--destroy', is_flag=True, required=False, help='Destroys cloud.')
@click.option('--no-destroy', is_flag=True, required=False, help='Keeps cloud.')
@click.option(
    '--cloud-tag', help='Tag under which all cloud resources are created.', required=False
)
@pass_config
def cloud(config: Config, **kwargs):
    """Launch/destroy a cloud compute instance.

    Launch/destroy cloud instances with Terraform based on Terraform files in tf_dir.
    """
    commands.cloud(config, **kwargs)
