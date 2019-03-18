variable "name" {}

variable "vpc_id" {}
variable "region" {}
variable "instance_type" {}


variable "public_key" {
  default = "~/.ssh/id_rsa.pub"
}

variable "private_key" {
  default = "~/.ssh/id_rsa"
}

variable "images" {
  type = "map"

  default = {
    eu-west-1 = "ami-088b2e2cc2498f3ca"
    eu-central-1 = "ami-055ab192b68ca4d2f"
  }
}

variable "ingress_cidr_blocks" {
  default = "0.0.0.0/0"
}

variable "s3_bucket" {}
