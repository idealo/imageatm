import os
import time
from pathlib import Path
from typing import Optional
from imageatm.handlers.utils import run_cmd
from imageatm.handlers.logger import get_logger


class AWS:
    """Cloud provider class that allows to run training on AWS.

    Cloud instances are created and destroyed using Terraform. The instance is provisioned
    with nvidia-docker and training is run in a Docker container using the public Docker
    image [idealo/tensorflow-image-atm:1.13.1](https://hub.docker.com/r/idealo/tensorflow-image-atm/tags).

    All commands on EC2 instance will be run via SSH.

    For training the local image and job directories will be synced with S3. After training
    the trained models will be synced with S3 and the local job directory.

    Attributes:
        tf_dir: Directory with Terraform files for AWS setup.
        region: AWS region [eu-west-1, eu-central-1].
        instance_type: AWS GPU instance type [g2.\*, p2.\*, p3.\*].
        vpc_id: AWS Virtual Private Cloud ID.
        s3_bucket: AWS S3 bucket where all training files will be stored (is not created by Terraform).
        job_dir: Job directory on local system (needed for logging).
        cloud_tag: Name under which all AWS resources will be set up.
    """

    def __init__(
        self,
        tf_dir: str,
        region: str,
        instance_type: str,
        vpc_id: str,
        s3_bucket: str,
        job_dir: str,
        cloud_tag: str,
        **kwargs
    ) -> None:
        self.tf_dir = tf_dir
        self.region = region
        self.instance_type = instance_type
        self.vpc_id = vpc_id
        self.s3_bucket = s3_bucket  # needed for IAM setup; bucket will not be created by terraform
        self.job_dir = os.path.abspath(job_dir)
        self.cloud_tag = cloud_tag

        self.image_dir: Optional[str] = None
        self.ssh: Optional[str] = None
        self.remote_workdir = '/home/ec2-user/image-atm'

        self._check_s3_prefix()

        self.logger = get_logger(__name__, Path(self.job_dir))

    def _check_s3_prefix(self):
        # ensure that s3 bucket prefix is correct
        self.s3_bucket_wo = self.s3_bucket.split('s3://')[-1]  # without s3:// prefix
        self.s3_bucket = 's3://' + self.s3_bucket_wo

    def _set_ssh(self):
        cmd = 'cd {} && terraform output public_ip'.format(self.tf_dir)
        output = run_cmd(cmd, logger=self.logger, return_output=True)
        self.ssh = 'ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa ec2-user@{}'.format(output)

    def _set_remote_dirs(self):
        cmd = '{} mkdir -p {}'.format(self.ssh, self.remote_workdir)
        run_cmd(cmd, logger=self.logger)

        self.remote_image_dir = os.path.join(self.remote_workdir, os.path.basename(self.image_dir))
        self.remote_job_dir = os.path.join(self.remote_workdir, os.path.basename(self.job_dir))

    def _set_s3_dirs(self):
        if 's3://' in self.image_dir:
            self.s3_image_dir = self.image_dir
        else:
            self.s3_image_dir = os.path.join(
                self.s3_bucket, 'image_dirs', os.path.basename(self.image_dir)
            )

        if 's3://' in self.job_dir:
            self.s3_job_dir = self.job_dir
        else:
            self.s3_job_dir = os.path.join(
                self.s3_bucket, 'job_dirs', os.path.basename(self.job_dir)
            )

    def _sync_local_s3(self):
        self.logger.info('Syncing files local <> s3...')
        self._set_s3_dirs()

        # only sync if image dir is local dir
        if 's3://' not in self.image_dir:
            cmd = 'aws s3 sync --quiet --exclude logs {} {}'.format(
                self.image_dir, self.s3_image_dir
            )
            run_cmd(cmd, logger=self.logger)

        # only sync if job dir is local dir
        if 's3://' not in self.job_dir:
            cmd = 'aws s3 sync --quiet --exclude logs {} {}'.format(self.job_dir, self.s3_job_dir)
            run_cmd(cmd, logger=self.logger)

    def _sync_s3_local(self):
        self.logger.info('Syncing files s3 <> local...')
        self._set_s3_dirs()

        # only sync if job dir is local dir
        if 's3://' not in self.job_dir:
            cmd = 'aws s3 sync --exclude logs --quiet {} {}'.format(self.s3_job_dir, self.job_dir)
            run_cmd(cmd, logger=self.logger, level='info')

    def _sync_remote_s3(self):
        self.logger.info('Syncing files remote <> s3...')
        self._set_s3_dirs()
        self._set_remote_dirs()

        cmd = '{} aws s3 sync --exclude logs --quiet {} {}'.format(
            self.ssh, self.remote_image_dir, self.s3_image_dir
        )
        run_cmd(cmd, logger=self.logger)

        cmd = '{} aws s3 sync --exclude logs --quiet {} {}'.format(
            self.ssh, self.remote_job_dir, self.s3_job_dir
        )
        run_cmd(cmd, logger=self.logger)

    def _sync_s3_remote(self):
        self.logger.info('Syncing files s3 <> remote...')
        self._set_s3_dirs()
        self._set_remote_dirs()

        cmd = '{} aws s3 sync --exclude logs --quiet {} {}'.format(
            self.ssh, self.s3_image_dir, self.remote_image_dir
        )
        run_cmd(cmd, logger=self.logger)

        cmd = '{} aws s3 sync --exclude logs --quiet {} {}'.format(
            self.ssh, self.s3_job_dir, self.remote_job_dir
        )
        run_cmd(cmd, logger=self.logger)

    def _launch_train_container(self, **kwargs):
        self.logger.info('Launching training container...')
        # training parameters are passed to container through environment variables
        envs = ['-e {}={}'.format(key, value) for key, value in kwargs.items() if value is not None]

        cmd = (
            '{} docker run -d -v {}:$WORKDIR/image_dir -v {}:$WORKDIR/job_dir {} '
            'idealo/tensorflow-image-atm:1.13.1'
        ).format(self.ssh, self.remote_image_dir, self.remote_job_dir, ' '.join(envs))

        run_cmd(cmd, logger=self.logger)

    def _stream_docker_logs(self):
        time.sleep(5)
        cmd = '{} docker ps -l -q'.format(self.ssh)
        output = run_cmd(cmd, logger=self.logger, return_output=True)

        cmd = '{} docker logs {} --follow'.format(self.ssh, output)
        run_cmd(cmd, logger=self.logger, level='info')

    def init(self):
        """Runs Terraform initialization."""
        self.logger.info('Running terraform init...')
        cmd = 'cd {} && terraform init'.format(self.tf_dir)
        run_cmd(cmd, logger=self.logger)

    def apply(self):
        """Runs Terraform apply."""
        self.logger.info('Running terraform apply...')
        cmd = (
            'cd {} && terraform apply -auto-approve -var "region={}" -var "instance_type={}" '
            '-var "vpc_id={}" -var "s3_bucket={}" -var "name={}"'
        ).format(
            self.tf_dir,
            self.region,
            self.instance_type,
            self.vpc_id,
            self.s3_bucket_wo,
            self.cloud_tag,
        )

        run_cmd(cmd, logger=self.logger)

        self._set_ssh()

    def train(self, image_dir: str = None, job_dir: str = None, **kwargs):
        """Runs training on EC2 instance.

        The following steps will be performed in sequence:
            - sync local image and job directory with S3
            - sync S3 with EC2 instance
            - launch Docker training container on EC2
            - sync EC2 with S3
            - sync S3 with local.

        Any of the pre-trained CNNs in Keras can be used.

        Args:
            image_dir: Directory with images used for training.
            job_dir: Directory with train_samples.json, val_samples.json,
                     and class_mapping.json.
            epochs_train_dense: Number of epochs to train dense layers.
            epochs_train_all: Number of epochs to train all layers.
            learning_rate_dense: Learning rate for dense training phase.
            self.learning_rate_all: Learning rate for all training phase.
            batch_size: Number of images per batch.
            dropout_rate: Fraction set randomly.
            base_model_name: Name of pretrained CNN.
        """
        self.logger.info('Setting up remote instance...')
        if image_dir is not None:
            self.image_dir = os.path.abspath(image_dir)

        self._sync_local_s3()
        self._sync_s3_remote()
        self._launch_train_container(**kwargs)
        self._stream_docker_logs()
        self._sync_remote_s3()
        self._sync_s3_local()

    def destroy(self):
        """Runs Terraform destroy."""
        self.logger.info('Running terraform destroy...')
        cmd = (
            'cd {} && terraform destroy -auto-approve -var "region={}" -var "instance_type={}" '
            '-var "vpc_id={}" -var "s3_bucket={}" -var "name={}"'
        ).format(
            self.tf_dir,
            self.region,
            self.instance_type,
            self.vpc_id,
            self.s3_bucket_wo,
            self.cloud_tag,
        )

        run_cmd(cmd, logger=self.logger)
