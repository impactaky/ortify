# ortify

Wrap PyTorch modules to run on ONNX Runtime.

## Installation

```bash
uv add ortify
```

## Development

```bash
uv sync --extra dev
uv run pre-commit install
uv run pre-commit run --all-files
uv run --extra dev ruff check .
```

## Usage

```python
from ortify import ortify, OrtifyArgs
import torch.nn as nn

model = nn.Linear(10, 5)
args = OrtifyArgs(ort_enabled=True)
wrapped = ortify(model, args)

output = wrapped(torch.randn(1, 10))  # Runs on ONNX Runtime
```

Pass ONNX Runtime session arguments through `onnxruntime_args` when needed:

```python
args = OrtifyArgs(
    ort_enabled=True,
    onnxruntime_args={"providers": ["CPUExecutionProvider"]},
)
```
