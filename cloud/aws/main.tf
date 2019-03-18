provider "aws" {
    region            = "${var.region}"
}

module "security" {
  source              = "./modules/security"
  name                = "${var.name}"
  vpc_id              = "${var.vpc_id}"
  ingress_cidr_blocks = "${var.ingress_cidr_blocks}"
}

module "key_pair" {
  source              = "./modules/key_pair"
  name                = "${var.name}"
  public_key          = "${var.public_key}"
}

module "ec2" {
  source              = "./modules/ec2"
  images              = "${var.images}"
  instance_type       = "${var.instance_type}"
  instance_profile    = "${module.iam.image_atm_instance_profile}"
  private_key         = "${var.private_key}"
  region              = "${var.region}"
  key_pair            = "${module.key_pair.image_atm_key_pair}"
  security_group      = "${module.security.image_atm_security_group}"
  name                = "${var.name}"
}

module "iam" {
  source              = "./modules/iam"
  s3_bucket           = "${var.s3_bucket}"
  name                = "${var.name}"
}
