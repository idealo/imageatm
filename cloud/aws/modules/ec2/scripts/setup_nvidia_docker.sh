#!/bin/bash
# source: https://docs.aws.amazon.com/batch/latest/userguide/batch-gpu-ami.html
# NB: cuda version from nvidia-smi and CUDA_VERSION don't match https://stackoverflow.com/questions/53422407/different-cuda-versions-shown-by-nvcc-and-nvidia-smi
# Install ecs-init, start docker, and install nvidia-docker 2
# sudo yum install -y ecs-init
# sudo service docker restart

DOCKER_VERSION=$(docker -v | awk '{ print $3 }' | cut -f1 -d"-")
DISTRIBUTION=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$DISTRIBUTION/nvidia-docker.repo | \
  sudo tee /etc/yum.repos.d/nvidia-docker.repo
PACKAGES=$(sudo yum search -y --showduplicates nvidia-docker2 nvidia-container-runtime | grep $DOCKER_VERSION | awk '{ print $1 }')
sudo yum install -y $PACKAGES
sudo pkill -SIGHUP dockerd

# Get CUDA version
CUDA_VERSION=$(cat /usr/local/cuda/version.txt | grep "^CUDA Version" | awk '{ print $3 }' | cut -f1-2 -d".")

# Run test container to verify installation
# sudo docker run --privileged --runtime=nvidia --rm nvidia/cuda:$CUDA_VERSION-base nvidia-smi

# Update Docker daemon.json to user nvidia-container-runtime by default
sudo tee /etc/docker/daemon.json <<EOF
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "default-runtime": "nvidia"
}
EOF

sudo service docker restart

# pull image-atm docker image
sudo docker pull idealo/tensorflow-image-atm:1.13.1

# add ec-user so that we can run docker commands without sudo
sudo usermod -a -G docker ec2-user

# TODO: might need to reboot instance as docker sometimes doesn't start
# https://github.com/hashicorp/terraform/issues/17844
