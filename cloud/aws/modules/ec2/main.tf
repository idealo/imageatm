resource "aws_instance" "image_atm" {
    ami                       = "${var.images[var.region]}"
    instance_type             = "${var.instance_type}"
    key_name                  = "${var.key_pair}"
    vpc_security_group_ids    = ["${var.security_group}"]
    iam_instance_profile      = "${var.instance_profile}"

    connection {
        user                  = "ec2-user"
        private_key           = "${file("${var.private_key}")}"
    }

    provisioner "remote-exec" {
        script                = "./modules/ec2/scripts/setup_nvidia_docker.sh"
    }

    tags {
        Name                  = "${var.name}"
    }
}
