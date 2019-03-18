# IAM role for EC2
data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    effect                            = "Allow"
    principals {
      type                            = "Service"
      identifiers                     = ["ec2.amazonaws.com"]
    }
    actions                           = ["sts:AssumeRole"]
  }
}

resource "aws_iam_role" "image_atm_ec2_instance_profile" {
  name                                = "${var.name}_EC2_InstanceProfile"
  assume_role_policy                  = "${data.aws_iam_policy_document.ec2_assume_role.json}"
}


# IAM role for S3
data "aws_iam_policy_document" "s3_read_write" {
  statement {
    effect                            = "Allow"
    actions                           = ["s3:ListBucket", "s3:GetObject", "s3:PutObject"]
    resources                         = ["arn:aws:s3:::${var.s3_bucket}", "arn:aws:s3:::${var.s3_bucket}/*"]
  }
}


resource "aws_iam_policy" "s3_policy" {
  name                                = "${var.name}_S3_Access"
  policy                              = "${data.aws_iam_policy_document.s3_read_write.json}"
}


# attach S3 policy to EC2 role
resource "aws_iam_policy_attachment" "image_atm_ec2_instance_profile" {
  name                                = "Attach S3 policy"
  roles                               = ["${aws_iam_role.image_atm_ec2_instance_profile.name}"]
  policy_arn                          = "${aws_iam_policy.s3_policy.arn}"
}


# tag IAM role to EC2 instance
resource "aws_iam_instance_profile" "image_atm_ec2_instance_profile" {
  name                                = "${aws_iam_role.image_atm_ec2_instance_profile.name}"
  role                                = "${aws_iam_role.image_atm_ec2_instance_profile.name}"
}
