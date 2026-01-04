# ortify

Wrap PyTorch modules to run on ONNX Runtime.

## Installation

```bash
pip install ortify
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
