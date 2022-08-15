import os
import sys

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(project_root)
    from src.config import get_terraform_variable

    mlflow_folder = os.path.join(project_root, "output", "mlruns").replace("\\", "/")
    os.makedirs(mlflow_folder, exist_ok=True)
    bucket_name = get_terraform_variable(
        os.path.join(project_root, "terraform", "variables.tf"), "mlflow_bucket"
    )
    bucket_path = f"gs://{bucket_name}"
    host = "0.0.0.0"
    mlflow_cmd = f"mlflow ui --backend-store-uri file://{mlflow_folder} --host {host} --default-artifact-root {bucket_path}"
    print(mlflow_cmd)
    os.system(mlflow_cmd)
