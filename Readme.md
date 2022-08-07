<a href="https://learn.hashicorp.com/tutorials/terraform/install-cli?in=terraform/gcp-get-started">Install terraform</a>
For windows the choco is required <a href="https://chocolatey.org/install#individual"> link </a>
Configure the terraform/main.tf file: change path to the google credentials, change the project name, change ssh private key path


terraform apply -auto-approve


Development plan:
1) download images with annotations
2) create dataset (in h5py format)
3) setup model registrty
4) write training code
5) create docker container for training
6) upload dataset and container to cloud
7) upload training package
8) call vertex ai training
9) download trained model
10) measure metrics
11) submit model to custom registry ? 


