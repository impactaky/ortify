from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn


@dataclass
class OrtifyArgs:
    """Configuration for ortify wrapper."""

    ort_enabled: bool = True
    export_path: Path | str | None = None
    opset_version: int = 17


class OrtifyWrapper(nn.Module):
    """Wrapper that runs a PyTorch module on ONNX Runtime."""

    def __init__(self, module: nn.Module, args: OrtifyArgs) -> None:
        super().__init__()
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected nn.Module, got {type(module).__name__}")

        self._module = module
        self._args = args
        self._session: ort.InferenceSession | None = None
        self._onnx_path: Path | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def _export_to_onnx(self, *args: torch.Tensor, **kwargs: Any) -> None:
        """Export the module to ONNX format."""
        if self._args.export_path is not None:
            self._onnx_path = Path(self._args.export_path)
        else:
            self._temp_dir = tempfile.TemporaryDirectory()
            self._onnx_path = Path(self._temp_dir.name) / "model.onnx"

        self._input_names = [f"input_{i}" for i in range(len(args))]

        self._module.eval()
        with torch.no_grad():
            torch.onnx.export(
                self._module,
                args,
                str(self._onnx_path),
                input_names=self._input_names,
                opset_version=self._args.opset_version,
                do_constant_folding=True,
            )

        self._session = ort.InferenceSession(
            str(self._onnx_path),
            providers=["CPUExecutionProvider"],
        )
        self._output_names = [o.name for o in self._session.get_outputs()]

    def forward(self, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Run forward pass, using ONNX Runtime if enabled."""
        if not self._args.ort_enabled:
            return self._module(*args, **kwargs)

        if self._session is None:
            self._export_to_onnx(*args, **kwargs)

        inputs = {
            name: arg.detach().cpu().numpy()
            for name, arg in zip(self._input_names, args)
        }

        outputs = self._session.run(self._output_names, inputs)

        device = args[0].device if args else torch.device("cpu")
        result = [torch.from_numpy(o).to(device) for o in outputs]

        if len(result) == 1:
            return result[0]
        return tuple(result)


def ortify(module: nn.Module, args: OrtifyArgs | None = None) -> OrtifyWrapper:
    """Wrap a PyTorch module to run on ONNX Runtime.

    Args:
        module: The PyTorch module to wrap.
        args: Configuration for the wrapper. If None, uses defaults.

    Returns:
        A wrapped module that runs on ONNX Runtime.

    Example:
        >>> model = nn.Linear(10, 5)
        >>> args = OrtifyArgs(ort_enabled=True)
        >>> wrapped = ortify(model, args)
        >>> output = wrapped(torch.randn(1, 10))
    """
    if args is None:
        args = OrtifyArgs()
    return OrtifyWrapper(module, args)
