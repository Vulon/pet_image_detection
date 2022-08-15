import json
import os
import sys

import yaml

project_root = os.environ["DVC_ROOT"]
sys.path.append(project_root)
from src.config import get_terraform_variable


def update_parameters(
    terraform_variables_file_path: str,
    parameters_file_path: str,
    train_package_path: str,
):
    mlflow_bucket = get_terraform_variable(
        terraform_variables_file_path, "mlflow_bucket"
    )
    with open(parameters_file_path, "r") as params_file:
        data = params_file.read()
    data = data.replace("<MLFLOW_BUCKET>", mlflow_bucket)
    with open(os.path.join(train_package_path, "params.yaml"), "w") as out_file:
        out_file.write(data)


def create_train_package():
    pass


if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    update_parameters(
        os.path.join(project_root, "terraform", "variables.tf"),
        os.path.join(project_root, "params.yaml"),
        os.path.join(project_root, "src", "train_package"),
    )
