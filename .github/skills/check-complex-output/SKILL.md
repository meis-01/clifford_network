---
name: check-complex-output
description: "Inspect PyTorch complex-valued neural networks and determine whether the final layer parameters, raw output, and loss-facing output are complex or real. Use for MNIST or other complex-network model checks, dtype debugging, and output-head verification."
argument-hint: "[model module or experiment path]"
user-invocable: true
---

# Check Complex Model Output

## When to Use

- Verify whether a model's last layer is complex-valued or real-valued.
- Distinguish complex parameters from the dtype returned by `forward()`.
- Check whether a classification head converts complex logits to real scores.
- Debug dtype mismatches in PyTorch complex-valued experiments.

## Procedure

1. Identify the model class used by the requested experiment or dataset.
2. Follow its constructor to the final module in the network.
3. Inspect the final layer's parameter declarations, including `weight` and `bias` dtypes.
4. Inspect every output path. Check both raw or explicitly complex methods such as `forward_complex()` and the public `forward()` method.
5. Run a minimal runtime check using the project's configured Python interpreter. Print:
   - final layer class,
   - final weight dtype,
   - final bias dtype when present,
   - raw output dtype,
   - public output dtype.
6. Report the result using separate labels:
   - **Final layer parameters:** complex or real.
   - **Raw final-layer output:** complex or real.
   - **Public model output:** complex or real.
7. If the public output is real but the final layer is complex, explain the conversion operation, such as `.abs()`, `.real`, magnitude, or a projection layer.

## MNIST Classifier Check

For this repository's FFT-MNIST classifier, use `input_size=784`, `num_classes=10`, and a valid configured activation. The project source is under `src/`, so run commands with `PYTHONPATH=src` from the project root. Prefer the active project interpreter; if `python` is unavailable, use its explicit executable path.

Example check:

```python
import torch
from clifford_network.models.complex_mlp import ComplexMLPClassifier

model = ComplexMLPClassifier(
    input_size=784,
    hidden_size=32,
    depth=1,
    num_classes=10,
    activation="split_tanh",
)
values = torch.randn(2, 784, dtype=torch.complex64)

print(type(model.network.head).__name__)
print(model.network.head.weight.dtype)
print(model.network.head.bias.dtype)
print(model.forward_complex(values).dtype)
print(model(values).dtype)
```

## Completion Criteria

- The answer names the exact final module and source path.
- Parameter dtype and output dtype are reported separately.
- The runtime check is performed when the environment permits it.
- Any complex-to-real conversion is identified explicitly.
- No conclusion is based only on the layer class name; the forward path is checked too.
