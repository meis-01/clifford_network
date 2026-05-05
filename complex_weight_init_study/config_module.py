import yaml
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    activation: str
    init_method: str
    in_features: Optional[int]
    hidden_size: int
    n_layers: int
    epochs: int
    batch_size: int
    device: str
    monitor_layers: List[int]
    output_dir: str
    dataset_name: str
    dataset_root: str


@dataclass
class ComparisonConfig:
    """Configuration for comparing multiple methods."""
    activations: List[str]
    init_methods: List[str]
    in_features: Optional[int]
    hidden_size: int
    n_layers: int
    epochs: int
    batch_size: int
    device: str
    monitor_layers: List[int]
    output_dir: str
    dataset_name: str
    dataset_root: str


def load_experiment_config(config_path: str) -> ExperimentConfig:
    """Load experiment configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate required fields
    required_fields = [
        'activation', 'init_method', 'in_features', 'hidden_size',
        'n_layers', 'epochs', 'batch_size', 'device', 'output_dir'
    ]

    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Required field '{field}' missing from config")

    # Set defaults for optional fields
    monitor_layers = config_dict.get('monitor_layers')
    if monitor_layers is None:
        # Default: monitor last layer and one in the middle
        n_layers = config_dict['n_layers']
        monitor_layers = [n_layers // 2, n_layers - 1] if n_layers > 1 else [n_layers - 1]

    dataset_name = config_dict.get('dataset_name', 'synthetic')
    dataset_root = config_dict.get('dataset_root', 'data')

    return ExperimentConfig(
        activation=config_dict['activation'],
        init_method=config_dict['init_method'],
        in_features=config_dict.get('in_features'),
        hidden_size=config_dict['hidden_size'],
        n_layers=config_dict['n_layers'],
        epochs=config_dict['epochs'],
        batch_size=config_dict['batch_size'],
        device=config_dict['device'],
        monitor_layers=monitor_layers,
        output_dir=config_dict['output_dir'],
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )


def load_comparison_config(config_path: str) -> ComparisonConfig:
    """Load comparison configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate required fields
    required_fields = [
        'activations', 'init_methods', 'in_features', 'hidden_size',
        'n_layers', 'epochs', 'batch_size', 'device', 'output_dir'
    ]

    for field in required_fields:
        if field not in config_dict:
            raise ValueError(f"Required field '{field}' missing from config")

    # Set defaults for optional fields
    monitor_layers = config_dict.get('monitor_layers')
    if monitor_layers is None:
        # Default: monitor last layer and one in the middle
        n_layers = config_dict['n_layers']
        monitor_layers = [n_layers // 2, n_layers - 1] if n_layers > 1 else [n_layers - 1]

    dataset_name = config_dict.get('dataset_name', 'synthetic')
    dataset_root = config_dict.get('dataset_root', 'data')

    return ComparisonConfig(
        activations=config_dict['activations'],
        init_methods=config_dict['init_methods'],
        in_features=config_dict.get('in_features'),
        hidden_size=config_dict['hidden_size'],
        n_layers=config_dict['n_layers'],
        epochs=config_dict['epochs'],
        batch_size=config_dict['batch_size'],
        device=config_dict['device'],
        monitor_layers=monitor_layers,
        output_dir=config_dict['output_dir'],
        dataset_name=dataset_name,
        dataset_root=dataset_root,
    )


def create_default_configs():
    """Create default configuration files."""

    # Single experiment config
    single_config = {
        'activation': 'complex_tanh',
        'init_method': 'structured_preserve',
        'in_features': 64,
        'hidden_size': 128,
        'n_layers': 20,
        'epochs': 50,
        'batch_size': 128,
        'device': 'cpu',
        'monitor_layers': [10, 19],  # Optional: will auto-set if not provided
        'dataset_name': 'synthetic',
        'dataset_root': 'data',
        'output_dir': 'single_experiment_results'
    }

    # Comparison config
    comparison_config = {
        'activations': [ 'complex_tanh'],
        'init_methods': [ 'trabelsi', 'structured_preserve'],
        'in_features': 64,
        'hidden_size': 128,
        'n_layers': 20,
        'epochs': 30,
        'batch_size': 128,
        'device': 'cpu',
        'monitor_layers': [10, 19],  # Optional: will auto-set if not provided
        'dataset_name': 'synthetic',
        'dataset_root': 'data',
        'output_dir': 'comparison_results'
    }

    # Deep network config
    deep_config = {
        'activations': ['complex_tanh'],
        'init_methods': ['xavier', 'he', 'structured_preserve'],
        'in_features': 64,
        'hidden_size': 128,
        'n_layers': 20,
        'epochs': 100,
        'batch_size': 256,
        'device': 'cpu',
        'monitor_layers': [5, 10, 15, 19],
        'dataset_name': 'synthetic',
        'dataset_root': 'data',
        'output_dir': 'deep_network_results'
    }

    configs_dir = 'configs'
    os.makedirs(configs_dir, exist_ok=True)

    # Write config files
    with open(os.path.join(configs_dir, 'single_experiment.yaml'), 'w') as f:
        yaml.dump(single_config, f, default_flow_style=False, sort_keys=False)

    with open(os.path.join(configs_dir, 'comparison.yaml'), 'w') as f:
        yaml.dump(comparison_config, f, default_flow_style=False, sort_keys=False)

    with open(os.path.join(configs_dir, 'deep_network.yaml'), 'w') as f:
        yaml.dump(deep_config, f, default_flow_style=False, sort_keys=False)

    print(f"Created default configuration files in {configs_dir}/")
    print("Available configs:")
    print("  - single_experiment.yaml: Single activation/init method combination")
    print("  - comparison.yaml: Compare multiple methods")
    print("  - deep_network.yaml: Deep network analysis")


if __name__ == "__main__":
    create_default_configs()