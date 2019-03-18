from imageatm.components import AWS


def run_cloud(
    provider: str,
    tf_dir: str,
    region: str,
    instance_type: str,
    vpc_id: str,
    bucket: str,
    destroy: bool,
    job_dir: str,
    cloud_tag: str,
    **kwargs
):
    if provider == 'aws':
        cloud = AWS(
            tf_dir=tf_dir,
            region=region,
            instance_type=instance_type,
            vpc_id=vpc_id,
            s3_bucket=bucket,
            job_dir=job_dir,
            cloud_tag=cloud_tag,
        )

    if destroy:
        cloud.destroy()

    else:
        cloud.init()
        cloud.apply()
