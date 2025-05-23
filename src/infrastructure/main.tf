# Terraform main.tf for LLM Financial Agent System on AWS

provider "aws" {
  region = "us-east-1"
}

variable "instance_type" {
  default = "g5.2xlarge"
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "main" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
}

resource "aws_security_group" "main" {
  vpc_id = aws_vpc.main.id

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "llm" {
  ami                    = "ami-0f9fc25dd2506cf6d" # Deep Learning AMI (Ubuntu 20.04) - update as needed
  instance_type          = var.instance_type
  subnet_id              = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.main.id]
  key_name               = "your-keypair"

  root_block_device {
    volume_size = 200
    volume_type = "gp3"
  }

  tags = {
    Name = "llm-inference"
  }
}

resource "aws_db_subnet_group" "main" {
  name       = "main"
  subnet_ids = [aws_subnet.main.id]
}

resource "aws_db_instance" "postgres" {
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "15.2"
  instance_class       = "db.t3.medium"
  name                 = "financedb"
  username             = "financeuser"
  password             = "StrongPassword123"
  parameter_group_name = "default.postgres15"
  skip_final_snapshot  = true
  vpc_security_group_ids = [aws_security_group.main.id]
  publicly_accessible  = true
  db_subnet_group_name = aws_db_subnet_group.main.name
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "main"
  subnet_ids = [aws_subnet.main.id]
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "finance-redis"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
  parameter_group_name = "default.redis7"
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.main.id]
}

resource "aws_efs_file_system" "shared" {
  creation_token = "finance-efs"
}

resource "aws_efs_mount_target" "efs_mount" {
  file_system_id  = aws_efs_file_system.shared.id
  subnet_id       = aws_subnet.main.id
  security_groups = [aws_security_group.main.id]
}
