#!/bin/bash
set -e

IMAGE_DIR=$WORKDIR/image_dir
JOB_DIR=$WORKDIR/job_dir

source ~/.venvs/image-atm/bin/activate

# start training
python -W ignore -m imageatm.scripts.run_training -j $JOB_DIR -i $IMAGE_DIR
