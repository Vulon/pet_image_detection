in Google console enable
Cloud storage API
Container registry API
Compute Engine API
Cloud RUN API

<a href="https://learn.hashicorp.com/tutorials/terraform/install-cli?in=terraform/gcp-get-started">Install terraform</a>
For windows the choco is required <a href="https://chocolatey.org/install#individual"> link </a>
Configure the terraform/main.tf file: change path to the google credentials, change the project name, change ssh private key path

In params.yaml change the project_name before calling dvc repro

terraform apply -auto-approve
