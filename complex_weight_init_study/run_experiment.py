import argparse
import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from complex_layers import ComplexMLP, get_activation_stats, get_gradient_norms
from initialization import initialize_model


def run_single_experiment(
    activation: str,
    init_method: str,
    in_features: int,
    hidden_size: int,
    n_layers: int,
    batch_size: int,
    device: str,
):
    model = ComplexMLP(
        in_features=in_features,
        hidden_size=hidden_size,
        n_layers=n_layers,
        activation=activation,
        use_output_layer=True,
    ).to(device)

    initialize_model(model, method=init_method)

    x = torch.randn(batch_size, in_features, dtype=torch.cfloat, device=device)

    output, pre_acts, post_acts = model(x)
    activation_stats = get_activation_stats(post_acts)

    loss = torch.mean(torch.abs(output) ** 2)
    loss.backward()
    grad_norms = get_gradient_norms(model)

    return {
        "activation": activation,
        "init_method": init_method,
        "activation_stats": activation_stats,
        "grad_norms": grad_norms,
        "output_norm": float(torch.abs(output).mean().cpu()),
        "loss": float(loss.cpu()),
    }


def plot_metrics(result, output_dir):
    prefix = f"{result['activation']}_{result['init_method']}"
    layer_index = list(range(1, len(result["activation_stats"]) + 1))

    means = [s["mean"] for s in result["activation_stats"]]
    stds = [s["std"] for s in result["activation_stats"]]
    min_vals = [s["min"] for s in result["activation_stats"]]
    max_vals = [s["max"] for s in result["activation_stats"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layer_index, means, marker="o", label="mean(|z|)")
    ax.plot(layer_index, stds, marker="s", label="std(|z|)")
    ax.set_title(f"Activation Magnitude Across Layers\n{result['activation']} / {result['init_method']}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Magnitude")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{prefix}_activation_stats.png"))
    plt.close(fig)

    if result["grad_norms"]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, len(result["grad_norms"]) + 1), result["grad_norms"], marker="o")
        ax.set_title(f"Gradient Norms by Parameter\n{result['activation']} / {result['init_method']}")
        ax.set_xlabel("Parameter Index")
        ax.set_ylabel("Gradient Norm")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{prefix}_grad_norms.png"))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run complex initialization experiments")
    parser.add_argument("--in-features", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--activations",
        nargs="+",
        default=["modrelu", "relu", "zrelu", "tanh"],
    )
    parser.add_argument(
        "--inits",
        nargs="+",
        default=["xavier", "he", "unitary", "random"],
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for activation in args.activations:
        for init_method in args.inits:
            print(f"Running activation={activation}, init={init_method}")
            result = run_single_experiment(
                activation=activation,
                init_method=init_method,
                in_features=args.in_features,
                hidden_size=args.hidden_size,
                n_layers=args.n_layers,
                batch_size=args.batch_size,
                device=args.device,
            )
            plot_metrics(result, args.output_dir)
            results.append(result)
            print(
                f"  output_norm={result['output_norm']:.4f}, loss={result['loss']:.4f}, "
                f"layers={len(result['activation_stats'])}"
            )

    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        for result in results:
            f.write(
                f"{result['activation']},{result['init_method']},"
                f"output_norm={result['output_norm']:.6f},loss={result['loss']:.6f},"
                f"mean_layer0={result['activation_stats'][0]['mean']:.6f},"
                f"mean_layerN={result['activation_stats'][-1]['mean']:.6f}\n"
            )
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
