<a href="https://learn.hashicorp.com/tutorials/terraform/install-cli?in=terraform/gcp-get-started">Install terraform</a>
For windows the choco is required <a href="https://chocolatey.org/install#individual"> link </a>
Configure the terraform/main.tf file: change path to the google credentials, change the project name, change VM IP. 


terraform apply -auto-approve

ssh -i .\id_rsa 34.76.252.190