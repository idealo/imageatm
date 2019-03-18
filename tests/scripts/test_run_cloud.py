import pytest
import shutil
from pathlib import Path
from imageatm.scripts import run_cloud
from imageatm.components.cloud import AWS


TEST_JOB_DIR = Path('./tests/data/test_train_job').resolve()
TEST_TF_DIR = 'test_tf_dir'
TEST_REGION = 'test_region'
TEST_INSTANCE_TYPE = 'test_instance_type'
TEST_VPC_ID = 'test_vpc_id'
TEST_S3_BUCKET = 'test_s3_bucket'
TEST_CLOUD_TAG = 'test_cloud_tag'


class TestRunEvaluation(object):
    def test_run_cloud_1(self, mocker):
        mp_init = mocker.patch('imageatm.components.cloud.AWS.init')
        mp_apply = mocker.patch('imageatm.components.cloud.AWS.apply')
        mp_destroy = mocker.patch('imageatm.components.cloud.AWS.destroy')
        mocker.patch('imageatm.components.cloud.AWS.__init__')
        AWS.__init__.return_value = None

        run_cloud(
            provider='aws',
            tf_dir=TEST_TF_DIR,
            region=TEST_REGION,
            instance_type=TEST_INSTANCE_TYPE,
            vpc_id=TEST_VPC_ID,
            bucket=TEST_S3_BUCKET,
            destroy=False,
            job_dir=TEST_JOB_DIR,
            cloud_tag=TEST_CLOUD_TAG,
            image_dir='test',
        )
        mp_init.assert_called_once()
        mp_apply.assert_called_once()
        mp_destroy.assert_not_called()

        AWS.__init__.assert_called_with(
            tf_dir=TEST_TF_DIR,
            region=TEST_REGION,
            instance_type=TEST_INSTANCE_TYPE,
            vpc_id=TEST_VPC_ID,
            s3_bucket=TEST_S3_BUCKET,
            job_dir=TEST_JOB_DIR,
            cloud_tag=TEST_CLOUD_TAG,
        )

    def test_run_cloud_2(self, mocker):
        mp_init = mocker.patch('imageatm.components.cloud.AWS.init')
        mp_apply = mocker.patch('imageatm.components.cloud.AWS.apply')
        mp_train = mocker.patch('imageatm.components.cloud.AWS.train')
        mp_destroy = mocker.patch('imageatm.components.cloud.AWS.destroy')
        mocker.patch('imageatm.components.cloud.AWS.__init__')
        AWS.__init__.return_value = None

        run_cloud(
            provider='aws',
            tf_dir=TEST_TF_DIR,
            region=TEST_REGION,
            instance_type=TEST_INSTANCE_TYPE,
            vpc_id=TEST_VPC_ID,
            bucket=TEST_S3_BUCKET,
            destroy=True,
            job_dir=TEST_JOB_DIR,
            cloud_tag=TEST_CLOUD_TAG,
        )
        mp_init.assert_not_called()
        mp_apply.assert_not_called()
        mp_destroy.assert_called_once()

        AWS.__init__.assert_called_with(
            tf_dir=TEST_TF_DIR,
            region=TEST_REGION,
            instance_type=TEST_INSTANCE_TYPE,
            vpc_id=TEST_VPC_ID,
            s3_bucket=TEST_S3_BUCKET,
            job_dir=TEST_JOB_DIR,
            cloud_tag=TEST_CLOUD_TAG,
        )
