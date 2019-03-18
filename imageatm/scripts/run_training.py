import os
import argparse
from imageatm.components import Training


def run_training(image_dir: str, job_dir: str, **kwargs):
    trainer = Training(image_dir=image_dir, job_dir=job_dir, **kwargs)
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job-dir', help='Directory with job files.', required=True)
    parser.add_argument('-i', '--image-dir', help='Directory with image files.', required=True)

    args = parser.parse_args()

    run_training(**{**os.environ, **args.__dict__})
