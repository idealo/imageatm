resource "aws_key_pair" "image_atm" {
    key_name                  = "${var.name}"
    public_key                = "${file("${var.public_key}")}"
}
