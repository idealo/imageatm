import pytest
from mock import call
from pathlib import Path
from yarl import URL
from imageatm.components.cloud import AWS

TEST_TF_DIR = Path('./tests/data/test_train_job').resolve()
TEST_REGION = 'test-region'
TEST_INSTANCE_TYPE = 'test_instance_type'
TEST_VPC_ID = 'test_vpc_id'
TEST_S3_BUCKET = URL('s3://test_s3_bucket')
TEST_JOB_DIR = Path('./tests/data/test_train_job').resolve()
TEST_CLOUD_TAG = 'test_cloud_tag'
TEST_IMG_DIR = Path('./tests/data/test_images').resolve()


@pytest.fixture(scope='class', autouse=True)
def tear_down(request):
    def remove_logs():
        (TEST_JOB_DIR / 'logs').unlink()

    request.addfinalizer(remove_logs)


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
        assert aws.remote_workdir == Path('/home/ec2-user/image-atm').resolve()
        mp__check_s3_prefix.assert_called_once()

    def test__check_s3_prefix(self):
        global aws

        aws.s3_bucket = TEST_S3_BUCKET
        aws._check_s3_prefix()
        assert aws.s3_bucket == URL('s3://test_s3_bucket')

        aws.s3_bucket = 'test_s3_bucket'
        aws._check_s3_prefix()
        assert aws.s3_bucket == URL('s3://test_s3_bucket')

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
        aws.job_dir = TEST_JOB_DIR

        assert not hasattr(aws, 'remote_image_dir')
        assert not hasattr(aws, 'remote_job_dir')

        aws._set_remote_dirs()
        assert hasattr(aws, 'remote_image_dir')
        assert hasattr(aws, 'remote_job_dir')
        assert aws.remote_image_dir == Path('/home/ec2-user/image-atm/test_images').resolve()
        assert aws.remote_job_dir == Path('/home/ec2-user/image-atm/test_train_job').resolve()

    def test__set_s3_dirs(self):
        aws.image_dir = TEST_IMG_DIR
        aws.job_dir = TEST_JOB_DIR

        assert not hasattr(aws, 's3_image_dir')
        assert not hasattr(aws, 's3_job_dir')

        aws._set_s3_dirs()

        assert aws.s3_image_dir == URL('s3://test_s3_bucket/image_dirs/test_images')
        assert aws.s3_job_dir == URL('s3://test_s3_bucket/job_dirs/test_train_job')

        aws.image_dir = URL('s3://test_s3_bucket/image_dirs/test_images2')
        aws.job_dir = URL('s3://test_s3_bucket/job_dirs/test_train_job2')

        aws._set_s3_dirs()

        assert aws.s3_image_dir == URL('s3://test_s3_bucket/image_dirs/test_images2')
        assert aws.s3_job_dir == URL('s3://test_s3_bucket/job_dirs/test_train_job2')

    def test__sync_local_s3_1(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.image_dir = TEST_IMG_DIR
        aws.job_dir = TEST_JOB_DIR
        calls = [
            call(
                'aws s3 sync --quiet --exclude logs {} {}'.format(
                    TEST_IMG_DIR, URL('s3://test_s3_bucket/image_dirs/test_images')
                ),
                logger=aws.logger,
            ),
            call(
                'aws s3 sync --quiet --exclude logs {} {}'.format(
                    TEST_JOB_DIR, URL('s3://test_s3_bucket/job_dirs/test_train_job')
                ),
                logger=aws.logger,
            ),
        ]

        aws._sync_local_s3()
        mp_run_cmd.assert_has_calls(calls)

    def test__sync_local_s3_2(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.image_dir = URL('s3://test_s3_bucket/image_dirs/test_images')
        aws.job_dir = URL('s3://test_s3_bucket/job_dirs/test_train_job')

        aws._sync_local_s3()
        mp_run_cmd.assert_not_called()

    def test__sync_s3_local_1(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.job_dir = TEST_JOB_DIR
        aws._sync_s3_local()
        mp_run_cmd.assert_called_with(
            'aws s3 sync --exclude logs --quiet {} {}'.format(
                URL('s3://test_s3_bucket/job_dirs/test_train_job'), TEST_JOB_DIR
            ),
            logger=aws.logger,
            level='info',
        )

    def test__sync_s3_local_2(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.job_dir = URL('s3://test_s3_bucket/job_dirs/test_train_job')
        aws._sync_s3_local()
        mp_run_cmd.assert_not_called()

    def test__sync_remote_s3_1(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.image_dir = TEST_IMG_DIR
        aws.job_dir = TEST_JOB_DIR
        aws.ssh = 'test_ssh'
        calls = [
            call(
                '{} aws s3 sync --exclude logs --quiet {} {}'.format(
                    'test_ssh',
                    Path('/home/ec2-user/image-atm/test_images'),
                    URL('s3://test_s3_bucket/image_dirs/test_images'),
                ),
                logger=aws.logger,
            ),
            call(
                '{} aws s3 sync --exclude logs --quiet {} {}'.format(
                    'test_ssh',
                    Path('/home/ec2-user/image-atm/test_train_job'),
                    URL('s3://test_s3_bucket/job_dirs/test_train_job'),
                ),
                logger=aws.logger,
            ),
        ]

        aws._sync_remote_s3()
        mp_run_cmd.assert_has_calls(calls)

    def test__sync_remote_s3_2(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.image_dir = URL('s3://test_s3_bucket/image_dirs/test_images')
        aws.job_dir = URL('s3://test_s3_bucket/job_dirs/test_train_job')
        aws.ssh = 'test_ssh'
        calls = [
            call(
                '{} aws s3 sync --exclude logs --quiet {} {}'.format(
                    'test_ssh',
                    Path('/home/ec2-user/image-atm/test_images'),
                    URL('s3://test_s3_bucket/image_dirs/test_images'),
                ),
                logger=aws.logger,
            ),
            call(
                '{} aws s3 sync --exclude logs --quiet {} {}'.format(
                    'test_ssh',
                    Path('/home/ec2-user/image-atm/test_train_job'),
                    URL('s3://test_s3_bucket/job_dirs/test_train_job'),
                ),
                logger=aws.logger,
            ),
        ]

        aws._sync_remote_s3()
        mp_run_cmd.assert_has_calls(calls)

    def test__sync_s3_remote_1(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.image_dir = TEST_IMG_DIR
        aws.job_dir = TEST_JOB_DIR
        aws.ssh = 'test_ssh'
        calls = [
            call(
                '{} aws s3 sync --exclude logs --quiet {} {}'.format(
                    'test_ssh',
                    URL('s3://test_s3_bucket/image_dirs/test_images'),
                    Path('/home/ec2-user/image-atm/test_images'),
                ),
                logger=aws.logger,
            ),
            call(
                '{} aws s3 sync --exclude logs --quiet {} {}'.format(
                    'test_ssh',
                    URL('s3://test_s3_bucket/job_dirs/test_train_job'),
                    Path('/home/ec2-user/image-atm/test_train_job'),
                ),
                logger=aws.logger,
            ),
        ]
        aws._sync_s3_remote()
        mp_run_cmd.assert_has_calls(calls)

    def test__sync_s3_remote_2(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.image_dir = URL('s3://test_s3_bucket/image_dirs/test_images')
        aws.job_dir = URL('s3://test_s3_bucket/job_dirs/test_train_job')
        aws.ssh = 'test_ssh'
        calls = [
            call(
                '{} aws s3 sync --exclude logs --quiet {} {}'.format(
                    'test_ssh',
                    URL('s3://test_s3_bucket/image_dirs/test_images'),
                    Path('/home/ec2-user/image-atm/test_images'),
                ),
                logger=aws.logger,
            ),
            call(
                '{} aws s3 sync --exclude logs --quiet {} {}'.format(
                    'test_ssh',
                    URL('s3://test_s3_bucket/job_dirs/test_train_job'),
                    Path('/home/ec2-user/image-atm/test_train_job'),
                ),
                logger=aws.logger,
            ),
        ]
        aws._sync_s3_remote()
        mp_run_cmd.assert_has_calls(calls)

    def test__launch_train_container(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        kwargs = {'key_1': 'test', 'key_2': 1, 'key_3': None}

        aws._launch_train_container(**kwargs)

        mp_run_cmd.assert_called_with(
            '{} docker run -d -v {}:$WORKDIR/image_dir -v {}:$WORKDIR/job_dir {} '
            'idealo/tensorflow-image-atm:1.13.1'.format(
                'test_ssh',
                Path('/home/ec2-user/image-atm/test_images'),
                Path('/home/ec2-user/image-atm/test_train_job'),
                '-e key_1=test -e key_2=1',
            ),
            logger=aws.logger,
        )

    def test__stream_docker_logs(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd', return_value='cmd_output')

        calls = [
            call('test_ssh docker ps -l -q', logger=aws.logger, return_output=True),
            call('test_ssh docker logs cmd_output --follow', logger=aws.logger, level='info'),
        ]
        aws._stream_docker_logs()
        mp_run_cmd.assert_has_calls(calls)

    def test_init(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.init()
        mp_run_cmd.assert_called_with(
            'cd {} && terraform init'.format(TEST_TF_DIR), logger=aws.logger
        )

    def test_apply(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        calls = [
            call(
                (
                    'cd {} && terraform apply -auto-approve -var "region={}" -var "instance_type={}" '
                    '-var "vpc_id={}" -var "s3_bucket={}" -var "name={}"'
                ).format(
                    TEST_TF_DIR,
                    TEST_REGION,
                    TEST_INSTANCE_TYPE,
                    TEST_VPC_ID,
                    URL('test_s3_bucket'),
                    TEST_CLOUD_TAG,
                ),
                logger=aws.logger,
            ),
            call(
                'cd {} && terraform output public_ip'.format(TEST_TF_DIR),
                logger=aws.logger,
                return_output=True,
            ),
        ]

        aws.apply()
        mp_run_cmd.assert_has_calls(calls)

    def test_train_1(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')
        mp__sync_local_s3 = mocker.patch('imageatm.components.cloud.AWS._sync_local_s3')
        mp__sync_s3_remote = mocker.patch('imageatm.components.cloud.AWS._sync_s3_remote')
        mp__launch_train_container = mocker.patch(
            'imageatm.components.cloud.AWS._launch_train_container'
        )
        mp__stream_docker_logs = mocker.patch('imageatm.components.cloud.AWS._stream_docker_logs')
        mp__sync_remote_s3 = mocker.patch('imageatm.components.cloud.AWS._sync_remote_s3')
        mp__sync_s3_local = mocker.patch('imageatm.components.cloud.AWS._sync_s3_local')

        aws.image_dir = TEST_IMG_DIR
        aws.job_dir = TEST_JOB_DIR

        aws.train()
        assert aws.image_dir == Path(TEST_IMG_DIR).resolve()
        mp__sync_local_s3.assert_called_once()
        mp__sync_s3_remote.assert_called_once()
        mp__launch_train_container.assert_called_once()
        mp__stream_docker_logs.assert_called_once()
        mp__sync_remote_s3.assert_called_once()
        mp__sync_s3_local.assert_called_once()

    def test_train_2(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')
        mp__sync_local_s3 = mocker.patch('imageatm.components.cloud.AWS._sync_local_s3')
        mp__sync_s3_remote = mocker.patch('imageatm.components.cloud.AWS._sync_s3_remote')
        mp__launch_train_container = mocker.patch(
            'imageatm.components.cloud.AWS._launch_train_container'
        )
        mp__stream_docker_logs = mocker.patch('imageatm.components.cloud.AWS._stream_docker_logs')
        mp__sync_remote_s3 = mocker.patch('imageatm.components.cloud.AWS._sync_remote_s3')
        mp__sync_s3_local = mocker.patch('imageatm.components.cloud.AWS._sync_s3_local')

        aws.train(image_dir='./tests/data/test_no_images')
        assert aws.image_dir == Path('./tests/data/test_no_images').resolve()
        mp__sync_local_s3.assert_called_once()
        mp__sync_s3_remote.assert_called_once()
        mp__launch_train_container.assert_called_once()
        mp__stream_docker_logs.assert_called_once()
        mp__sync_remote_s3.assert_called_once()
        mp__sync_s3_local.assert_called_once()

    def test_destroy(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')

        aws.destroy()
        mp_run_cmd.assert_called_with(
            (
                'cd {} && terraform destroy -auto-approve -var "region={}" -var "instance_type={}" '
                '-var "vpc_id={}" -var "s3_bucket={}" -var "name={}"'
            ).format(
                TEST_TF_DIR,
                TEST_REGION,
                TEST_INSTANCE_TYPE,
                TEST_VPC_ID,
                URL('test_s3_bucket'),
                TEST_CLOUD_TAG,
            ),
            logger=aws.logger,
        )

    def test_integration(self, mocker):
        mp_run_cmd = mocker.patch('imageatm.components.cloud.run_cmd')
        mp__sync_local_s3 = mocker.patch('imageatm.components.cloud.AWS._sync_local_s3')
        mp__sync_s3_remote = mocker.patch('imageatm.components.cloud.AWS._sync_s3_remote')
        mp__launch_train_container = mocker.patch(
            'imageatm.components.cloud.AWS._launch_train_container'
        )
        mp__stream_docker_logs = mocker.patch('imageatm.components.cloud.AWS._stream_docker_logs')
        mp__sync_remote_s3 = mocker.patch('imageatm.components.cloud.AWS._sync_remote_s3')
        mp__sync_s3_local = mocker.patch('imageatm.components.cloud.AWS._sync_s3_local')

        aws2 = AWS(
            tf_dir=TEST_TF_DIR,
            region=TEST_REGION,
            instance_type=TEST_INSTANCE_TYPE,
            vpc_id=TEST_VPC_ID,
            s3_bucket=TEST_S3_BUCKET,
            job_dir=TEST_JOB_DIR,
            cloud_tag=TEST_CLOUD_TAG,
        )

        aws2.init()
        aws2.apply()

        NEW_JOB_DIR = Path('./tests/data/test_train_job_copy').resolve()

        aws2.train(job_dir=NEW_JOB_DIR)
        aws2.destroy()

        assert aws2.job_dir == NEW_JOB_DIR
        mp__sync_local_s3.assert_called_once()
        mp__sync_s3_remote.assert_called_once()
        mp__launch_train_container.assert_called_once()
        mp__stream_docker_logs.assert_called_once()
        mp__sync_remote_s3.assert_called_once()
        mp__sync_s3_local.assert_called_once()
