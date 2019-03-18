output "public_ip" {
    value             = "${module.ec2.image_atm_public_ip}"
}
