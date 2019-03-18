import pytest
import shutil
from pathlib import Path
from imageatm.scripts import run_training_cloud
from imageatm.components.cloud import AWS


TEST_JOB_DIR = Path('./tests/data/test_train_job').resolve()
TEST_TF_DIR = 'test_tf_dir'
TEST_REGION = 'test_region'
TEST_INSTANCE_TYPE = 'test_instance_type'
TEST_VPC_ID = 'test_vpc_id'
TEST_S3_BUCKET = 'test_s3_bucket'
TEST_CLOUD_TAG = 'test_cloud_tag'

TEST_IMAGE_DIR = Path('./tests/data/test_images').resolve()
TEST_BATCH_SIZE = 64
TEST_DROPOUT_RATE = 0.75
TEST_BASE_MODEL_NAME = 'MobileNet'
TEST_LOSS = 'test_loss'
TEST_EPOCHS_TRAIN_DENSE = 2
TEST_EPOCHS_TRAIN_ALL = 10
TEST_LEARNING_RATE_DENSE = 0.001
TEST_LEARNING_RATE_ALL = 0.0001


class TestRunEvaluation(object):
    def test_run_training_cloud_1(self, mocker):
        mp_init = mocker.patch('imageatm.components.cloud.AWS.init')
        mp_apply = mocker.patch('imageatm.components.cloud.AWS.apply')
        mp_train = mocker.patch('imageatm.components.cloud.AWS.train')
        mp_destroy = mocker.patch('imageatm.components.cloud.AWS.destroy')
        mocker.patch('imageatm.components.cloud.AWS.__init__')
        AWS.__init__.return_value = None

        run_training_cloud(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            provider='aws',
            tf_dir=TEST_TF_DIR,
            region=TEST_REGION,
            instance_type=TEST_INSTANCE_TYPE,
            vpc_id=TEST_VPC_ID,
            bucket=TEST_S3_BUCKET,
            destroy=False,
            cloud_tag=TEST_CLOUD_TAG,
        )
        mp_init.assert_called_once()
        mp_apply.assert_called_once()
        mp_train.assert_called_with(job_dir=TEST_JOB_DIR, image_dir=TEST_IMAGE_DIR)
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

    def test_run_training_cloud_2(self, mocker):
        mp_init = mocker.patch('imageatm.components.cloud.AWS.init')
        mp_apply = mocker.patch('imageatm.components.cloud.AWS.apply')
        mp_train = mocker.patch('imageatm.components.cloud.AWS.train')
        mp_destroy = mocker.patch('imageatm.components.cloud.AWS.destroy')
        mocker.patch('imageatm.components.cloud.AWS.__init__')
        AWS.__init__.return_value = None

        run_training_cloud(
            image_dir=TEST_IMAGE_DIR,
            job_dir=TEST_JOB_DIR,
            provider='aws',
            tf_dir=TEST_TF_DIR,
            region=TEST_REGION,
            instance_type=TEST_INSTANCE_TYPE,
            vpc_id=TEST_VPC_ID,
            bucket=TEST_S3_BUCKET,
            destroy=True,
            cloud_tag=TEST_CLOUD_TAG,
            epochs_train_dense=TEST_EPOCHS_TRAIN_DENSE,
            epochs_train_all=TEST_EPOCHS_TRAIN_ALL,
            learning_rate_dense=TEST_LEARNING_RATE_DENSE,
            learning_rate_all=TEST_LEARNING_RATE_ALL,
            batch_size=TEST_BATCH_SIZE,
            dropout_rate=TEST_DROPOUT_RATE,
            base_model_name=TEST_BASE_MODEL_NAME,
            loss=TEST_LOSS,
        )
        mp_init.assert_called_once()
        mp_apply.assert_called_once()
        mp_train.assert_called_with(
            job_dir=TEST_JOB_DIR,
            image_dir=TEST_IMAGE_DIR,
            epochs_train_dense=TEST_EPOCHS_TRAIN_DENSE,
            epochs_train_all=TEST_EPOCHS_TRAIN_ALL,
            learning_rate_dense=TEST_LEARNING_RATE_DENSE,
            learning_rate_all=TEST_LEARNING_RATE_ALL,
            batch_size=TEST_BATCH_SIZE,
            dropout_rate=TEST_DROPOUT_RATE,
            base_model_name=TEST_BASE_MODEL_NAME,
            loss=TEST_LOSS,
        )
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
