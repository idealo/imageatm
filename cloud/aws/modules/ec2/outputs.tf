
output "image_atm_public_ip" {
    value = "${element(concat(aws_instance.image_atm.*.public_ip, list("")), 0)}"
    # https://github.com/hashicorp/terraform/issues/17862
    # value = "${aws_instance.image_atm.public_ip}"
}
