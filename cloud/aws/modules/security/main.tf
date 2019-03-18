resource "aws_security_group" "image_atm" {
  name                   = "${var.name}"
  description            = "Security group for Image-ATM."
  vpc_id                 = "${var.vpc_id}"
  revoke_rules_on_delete = true

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["${var.ingress_cidr_blocks}"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags {
    Name = "Image-ATM"
  }
}
