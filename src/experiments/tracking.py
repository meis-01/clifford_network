from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import mlflow
except ImportError:  # pragma: no cover - optional dependency
    mlflow = None


@dataclass
class ExperimentTracker:
    experiment_name: str = "fastmri-prostate"
    enabled: bool = False
    tags: dict[str, str] = field(default_factory=dict)

    def start_run(self, run_name: str | None = None) -> None:
        if not self.enabled or mlflow is None:
            return
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run(run_name=run_name)
        if self.tags:
            mlflow.set_tags(self.tags)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self.enabled or mlflow is None:
            return
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        if not self.enabled or mlflow is None:
            return
        mlflow.log_metrics(metrics, step=step)

    def end_run(self) -> None:
        if not self.enabled or mlflow is None:
            return
        mlflow.end_run()
