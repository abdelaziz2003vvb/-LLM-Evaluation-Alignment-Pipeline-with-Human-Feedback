import mlflow
from typing import Dict, Any


class MLflowTracker:
    def __init__(self, tracking_uri: str, experiment_name: str):
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str = None):
        return mlflow.start_run(run_name=run_name)

    def log_metrics(self, metrics: Dict[str, Any], step: int = None):
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v), step=step)

    def log_params(self, params: Dict[str, Any]):
        for k, v in params.items():
            mlflow.log_param(k, str(v))

    def log_artifact(self, path: str):
        mlflow.log_artifact(path)

    def register_model(self, local_path: str, model_name: str):
        mlflow.pyfunc.log_model(artifact_path=model_name, python_model=None, artifacts={"model": local_path})
