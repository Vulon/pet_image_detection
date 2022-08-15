import os
import sys

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_root)
    from src.config import get_config_from_yaml, get_terraform_variable

    config = get_config_from_yaml(os.path.join(project_root, "params.yaml"))

    mlflow_folder = os.path.join(project_root, "output", "mlruns").replace("\\", "/")
    os.makedirs(mlflow_folder, exist_ok=True)
    bucket_name = get_terraform_variable(
        os.path.join(project_root, "terraform", "variables.tf"), "mlflow_bucket"
    )
    bucket_path = f"gs://{bucket_name}"

    host = config.mlflow.host
    port = config.mlflow.port
    mlflow_cmd = f"mlflow ui --backend-store-uri sqlite:///{mlflow_folder}/mlflow.db --host {host} --port {port} --default-artifact-root {bucket_path}"
    print(mlflow_cmd)
    os.system(mlflow_cmd)
