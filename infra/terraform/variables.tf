variable "aws_region" {
  default = "us-east-1"
}

variable "cluster_name" {
  default = "mra-cluster"
}

variable "subnet_ids" {
  type = list(string)
  default = ["subnet-xyz1", "subnet-xyz2"]
}
