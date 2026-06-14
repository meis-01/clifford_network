import os
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

try:
    from torchvision.datasets import MNIST, CIFAR10
    import torchvision.transforms as T
except ImportError:  # pragma: no cover
    MNIST = None
    CIFAR10 = None
    T = None


def _fft_image_to_complex(image_tensor: torch.Tensor) -> torch.Tensor:
    """Convert a real image tensor to a complex-valued FFT feature vector."""
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    if image_tensor.ndim != 3:
        raise ValueError("Input image tensor must have shape [C,H,W] or [H,W].")

    image_tensor = image_tensor.to(torch.float32)
    fft_complex = torch.fft.fft2(image_tensor, norm="ortho")
    return fft_complex.flatten()


class ComplexDataset(Dataset):
    """Dataset for complex-valued inputs and targets."""

    def __init__(
        self,
        n_samples: int = 0,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        inputs: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        if inputs is not None:
            if inputs.dtype != torch.cfloat:
                raise ValueError("`inputs` must be a complex tensor with dtype=torch.cfloat")
            self.inputs = inputs
            self.n_samples = inputs.shape[0]
            self.in_features = inputs.shape[1]
        else:
            if in_features is None:
                raise ValueError("`in_features` must be specified when no `inputs` are provided")
            self.n_samples = n_samples
            self.in_features = in_features
            self.inputs = torch.randn(n_samples, in_features, dtype=torch.cfloat)

        self.out_features = out_features or self.in_features
        self.targets = self._generate_targets()

    def _generate_targets(self) -> torch.Tensor:
        """Generate targets as a linear transformation with noise."""
        W = torch.randn(self.out_features, self.in_features, dtype=torch.cfloat)
        targets = self.inputs @ W.T
        noise = torch.randn_like(targets) * 0.1
        return targets + noise

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ComplexImageFFTDataset(ComplexDataset):
    """Complex-valued dataset built from MNIST or CIFAR10 images using FFT."""

    SUPPORTED_DATASETS = {"mnist", "cifar10"}

    def __init__(
        self,
        dataset_name: str,
        root: str = "data",
        train: bool = True,
        n_samples: Optional[int] = None,
        download: bool = True,
        out_features: Optional[int] = None,
    ):
        if T is None or MNIST is None or CIFAR10 is None:
            raise ImportError(
                "To use FFT image datasets you must install torchvision: `pip install torchvision`"
            )

        dataset_name = dataset_name.lower()
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        transform = T.ToTensor()
        if dataset_name == "mnist":
            base_dataset = MNIST(root, train=train, download=download, transform=transform)
        else:
            base_dataset = CIFAR10(root, train=train, download=download, transform=transform)

        n_samples = min(n_samples or len(base_dataset), len(base_dataset))
        inputs = []
        for i in range(n_samples):
            image, _ = base_dataset[i]
            inputs.append(_fft_image_to_complex(image))

        inputs = torch.stack(inputs, dim=0)
        super().__init__(n_samples=inputs.shape[0], inputs=inputs, out_features=out_features)


def create_data_loaders(
    train_samples: int = 10000,
    val_samples: int = 2000,
    batch_size: int = 128,
    in_features: Optional[int] = None,
    out_features: Optional[int] = None,
    dataset_name: str = "synthetic",
    dataset_root: str = "data",
):
    """Create train and validation data loaders for synthetic or FFT image datasets."""
    dataset_name = dataset_name.lower()

    if dataset_name in {"mnist", "cifar10"}:
        train_dataset = ComplexImageFFTDataset(
            dataset_name=dataset_name,
            root=dataset_root,
            train=True,
            n_samples=train_samples,
            download=True,
            out_features=out_features,
        )
        val_dataset = ComplexImageFFTDataset(
            dataset_name=dataset_name,
            root=dataset_root,
            train=False,
            n_samples=val_samples,
            download=True,
            out_features=out_features,
        )
    else:
        if in_features is None:
            raise ValueError("in_features must be set for synthetic datasets")
        train_dataset = ComplexDataset(train_samples, in_features, out_features)
        val_dataset = ComplexDataset(val_samples, in_features, out_features)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


class ComplexRegressionDataset(Dataset):
    """Dataset for complex regression tasks."""

    def __init__(self, n_samples: int, in_features: int, hidden_dim: int = 32):
        super().__init__()
        self.n_samples = n_samples
        self.in_features = in_features

        # Generate complex inputs
        self.inputs = torch.randn(n_samples, in_features, dtype=torch.cfloat)

        # Generate targets through a complex network
        self.targets = self._generate_complex_targets(hidden_dim)

    def _generate_complex_targets(self, hidden_dim: int) -> torch.Tensor:
        """Generate targets by passing through a random complex network."""
        W1 = torch.randn(hidden_dim, self.in_features, dtype=torch.cfloat)
        W2 = torch.randn(1, hidden_dim, dtype=torch.cfloat)  # Output single complex value

        hidden = torch.complex(
            torch.relu((self.inputs @ W1.T).real),
            torch.relu((self.inputs @ W1.T).imag)
        )
        targets = hidden @ W2.T

        noise = torch.randn_like(targets) * 0.05
        return targets + noise

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
