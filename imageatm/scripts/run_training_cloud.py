from imageatm.components import AWS


def run_training_cloud(
    image_dir: str,
    job_dir: str,
    provider: str,
    tf_dir: str,
    region: str,
    instance_type: str,
    vpc_id: str,
    bucket: str,
    destroy: bool,
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

    cloud.init()
    cloud.apply()
    cloud.train(image_dir=image_dir, job_dir=job_dir, **kwargs)

    if destroy:
        cloud.destroy()
