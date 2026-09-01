"""Config-driven model construction.

The registry chooses the appropriate complex model family for the experiment
task and fills in dimensions discovered from the selected dataset.
"""

from __future__ import annotations

from torch import nn

from clifford_network.models.complex_mlp import ComplexMLPAutoencoder, ComplexMLPClassifier, ComplexMLPRegressor


def build_model(config: dict, *, input_size: int, num_classes: int | None, task: str, depth: int) -> nn.Module:
    """Build the configured classifier or autoencoder for a dataset spec."""
    model_config = config.get("model", {})
    name = model_config.get("name", "complex_mlp").lower()
    hidden_size = int(model_config.get("hidden_size", 128))
    activation = model_config.get("activation", "split_tanh")

    if task == "classification":
        if name not in {"complex_mlp", "complex_classifier"}:
            raise ValueError(f"Unsupported classification model '{name}'.")
        if num_classes is None:
            raise ValueError("Classification models require num_classes.")
        return ComplexMLPClassifier(
            input_size=input_size,
            hidden_size=hidden_size,
            depth=depth,
            num_classes=num_classes,
            activation=activation,
        )
    if task == "regression":
        if name not in {
            "complex_mlp",
            "complex_regressor",
        }:
            raise ValueError(
                f"Unsupported regression model '{name}'."
            )

        return ComplexMLPRegressor(
            input_size=input_size,
            hidden_size=hidden_size,
            depth=depth,
            activation=activation,
            output_size=1,
        )

    if task == "autoencoder":
        if name not in {"complex_mlp_autoencoder", "complex_autoencoder"}:
            raise ValueError(f"Unsupported autoencoder model '{name}'.")
        return ComplexMLPAutoencoder(
            input_size=input_size,
            hidden_size=hidden_size,
            depth=depth,
            activation=activation,
            latent_size=model_config.get("latent_size"),
        )

    raise ValueError(f"Unsupported task '{task}'.")
