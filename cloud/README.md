# Image-ATM Terraform files

For each cloud provider that is supported in `imageatm/components/cloud.py` there is a corresponding set of Terraform
configuration files in this directory.

The Image-ATM cloud component will run Terraform *apply* and *destroy* commands based on the cloud configurations
specified by the user.

## Setup cloud user rights
### AWS

1. Assign the following **IAM roles** to your AWS user account

* iam role:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "Stmt1508403745000",
            "Effect": "Allow",
            "Action": [
                "iam:CreatePolicy",
                "iam:CreateRole",
                "iam:GetPolicy",
                "iam:GetRole",
                "iam:GetPolicyVersion",
                "iam:CreateInstanceProfile",
                "iam:AttachRolePolicy",
                "iam:ListRolePolicies",
                "iam:GetInstanceProfile",
                "iam:ListEntitiesForPolicy",
                "iam:ListPolicyVersions",
                "iam:CreatePolicyVersion",
                "iam:RemoveRoleFromInstanceProfile",
                "iam:DetachRolePolicy",
                "iam:DeleteInstanceProfile",
                "iam:DeletePolicyVersion",
                "iam:ListInstanceProfilesForRole",
                "iam:DeletePolicy",
                "iam:DeleteRole",
                "iam:AddRoleToInstanceProfile",
                "iam:PassRole"
            ],
            "Resource": [
                "*"
            ]
        }
    ]
}
```

* ec2 role:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": "ec2:*",
            "Effect": "Allow",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "elasticloadbalancing:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "cloudwatch:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "autoscaling:*",
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": "iam:CreateServiceLinkedRole",
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "iam:AWSServiceName": [
                        "autoscaling.amazonaws.com",
                        "ec2scheduled.amazonaws.com",
                        "elasticloadbalancing.amazonaws.com",
                        "spot.amazonaws.com",
                        "spotfleet.amazonaws.com",
                        "transitgateway.amazonaws.com"
                    ]
                }
            }
        }
    ]
}
```

* s3 role:
``` json
{
   "Version": "2012-10-17",
   "Statement": [
       {
           "Effect": "Allow",
           "Action": "s3:*",
           "Resource": "*"
       }
   ]
}
```

2. Configure your AWS client with `aws configure`

3. Init and plan the Terraform configurations

```bash
cd aws
terraform init
terraform plan
```

You will be asked to fill in the following variables:
  - region: AWS region. Currently supported [eu-west-1, eu-central-1]
  - instance_type: EC2 instance type. Currently supported [g2.\*, p2.\*, p3.\*]
  - vpc_id: AWS Virtual Private Cloud ID
  - s3_bucket: AWS S3 bucket where all training files will be stored (is not created by Terraform).
  - name: Name under which all AWS resources will be set up.

You should be ready to use the Image-ATM cloud component with the `provider='aws'` configuration now.
