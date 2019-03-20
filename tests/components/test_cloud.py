import os
import pytest
import shutil
import numpy.testing as npt
from pathlib import Path
from imageatm.components.cloud import AWS

TEST_TF_DIR = os.path.abspath('./tests/data/test_train_job')
TEST_REGION = 'test-region'
TEST_INSTANCE_TYPE = 'test_instance_type'
TEST_VPC_ID = 'test_vpc_id'
TEST_S3_BUCKET = 's3://test_s3_bucket'
TEST_JOB_DIR = os.path.abspath('./tests/data/test_train_job')
TEST_CLOUD_TAG = 'test_cloud_tag'
TEST_IMG_DIR = os.path.abspath('./tests/data/test_images')


class TestAWS(object):

    aws = None

    def test__init__(self, mocker):
        mp__check_s3_prefix = mocker.patch('imageatm.components.cloud.AWS._check_s3_prefix')

        global aws
        aws = AWS(
            tf_dir=TEST_TF_DIR,
            region=TEST_REGION,
            instance_type=TEST_INSTANCE_TYPE,
            vpc_id=TEST_VPC_ID,
            s3_bucket=TEST_S3_BUCKET,
            job_dir=TEST_JOB_DIR,
            cloud_tag=TEST_CLOUD_TAG,
        )

        assert aws.tf_dir == TEST_TF_DIR
        assert aws.region == TEST_REGION
        assert aws.instance_type == TEST_INSTANCE_TYPE
        assert aws.vpc_id == TEST_VPC_ID
        assert aws.s3_bucket == TEST_S3_BUCKET
        assert aws.job_dir == Path(TEST_JOB_DIR)
        assert aws.cloud_tag == TEST_CLOUD_TAG
        assert aws.remote_workdir == '/home/ec2-user/image-atm'
        mp__check_s3_prefix.assert_called_once()

    def test__check_s3_prefix(self):
        global aws

        aws.s3_bucket = TEST_S3_BUCKET
        aws._check_s3_prefix()
        assert aws.s3_bucket == 's3://test_s3_bucket'

        aws.s3_bucket = 'test_s3_bucket'
        aws._check_s3_prefix()
        assert aws.s3_bucket == 's3://test_s3_bucket'

    def test__set_ssh(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        assert aws.ssh is None
        aws._set_ssh()
        assert aws.ssh is not None
        mp_run_cmd.assert_called_with(
            'cd {} && terraform output public_ip'.format(TEST_TF_DIR),
            logger=aws.logger,
            return_output=True,
        )

    def test__set_remote_dirs(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        global aws
        aws.image_dir = TEST_IMG_DIR

        assert not hasattr(aws, 'remote_image_dir')
        assert not hasattr(aws, 'remote_job_dir')

        aws._set_remote_dirs()
        assert hasattr(aws, 'remote_image_dir')
        assert hasattr(aws, 'remote_job_dir')

    def test__set_s3_dirs(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')
        pass

    def test__sync_local_s3(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')
        pass

    def test__sync_s3_local(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')
        pass

    def test__sync_remote_s3(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')
        pass

    def test__sync_s3_remote(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')
        pass

    def test__launch_train_container(self, mocker):
        pass

    def test__stream_docker_logs(self, mocker):
        pass
