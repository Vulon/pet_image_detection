<h2>Description</h2>
This project trains and deploys image segmentation neural network for cats and dogs images.
It is based on pretrained deeplabv3 model and finetuned on COCO datset 2017.

Workflow is specified in dvc.yaml file. <p>
First stage downloads annotations from COCO website

Annotations are filtered, and only data with pet annotations remain.

At third stage the images are downloaded and rescaled to the 512 x 512 size.
All images are stored in HDF5 format.
Annotations are parsed and transformed to binary masks that are stored in HDF5 format as well.

Forth stage trains the model.

Then the docker container for deployment is built.
It gets uploaded to Google Container Registry and served in Google Cloud Run afterwards.


<h2>Steps to reproduce</h2>
Create Google Cloud Platform project

in Google console enable
<ol>
<li>Cloud storage API</li>
<li>Container registry API</li>
<li>Compute Engine API</li>
<li>Cloud RUN API</li>
</ol>
Create service account for Mlflow and Terraform. Download json credentials to the keys folder.

Change key name in params.yaml


<a href="https://learn.hashicorp.com/tutorials/terraform/install-cli?in=terraform/gcp-get-started">Install terraform</a>

For windows the choco is required <a href="https://chocolatey.org/install#individual"> link. </a>

Configure the terraform/main.tf file: change path to the google credentials, change the project name, change ssh private key path

In params.yaml change the project_name before calling dvc repro


To provision VM and prepare project call > <b>terraform apply -auto-approve</b>

This process might take some time.

Once the VM is prepared you can connect to it with ssh.
IP can be found in <a href="https://console.cloud.google.com/compute/">google console</a>

You can start the mlflow server by calling <i>pipenv run python scripts/start_mlflow.py</i>

To train a model you can type <i>pipenv run dvc repro</i>


<h2>Tech stack</h2>

<ul>
<li>language: python 3.10</li>
<li>Cloud: Google Cloud Platform</li>
<li>IaC: Terraform</li>
<li>Experiment tracking: mlflow + dvc</li>
<li>model registry: mlflow</li>
<li>Workflow orchestration: dvc</li>
<li>Model deployment: Model containerized with Fast API server</li>
<li>Unit tests: unittest</li>
<li>Formatter: black and isort</li>
<li>pre-commit hooks: pre-commit</li>
<li>Makefile: file with bash commands to setup VM</li>

</ul>
