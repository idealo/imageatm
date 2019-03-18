output "image_atm_instance_profile" {
  value = "${aws_iam_role.image_atm_ec2_instance_profile.name}"
}
