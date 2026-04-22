import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from ortify import OrtifyArgs, OrtifyWrapper, ortify


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestOrtify:
    def test_basic_wrapping(self):
        model = SimpleModel()
        wrapped = ortify(model)
        assert isinstance(wrapped, OrtifyWrapper)

    def test_invalid_module_raises_typeerror(self):
        with pytest.raises(TypeError):
            ortify("not a module")

    def test_default_args(self):
        model = SimpleModel()
        wrapped = ortify(model)
        assert wrapped._args.ort_enabled is True
        assert wrapped._args.opset_version == 17
        assert wrapped._args.onnxruntime_args == {}

    def test_custom_args(self):
        model = SimpleModel()
        args = OrtifyArgs(
            ort_enabled=False,
            opset_version=14,
            onnxruntime_args={"providers": ["CPUExecutionProvider"]},
        )
        wrapped = ortify(model, args)
        assert wrapped._args.ort_enabled is False
        assert wrapped._args.opset_version == 14
        assert wrapped._args.onnxruntime_args == {"providers": ["CPUExecutionProvider"]}


class TestOrtifyWrapper:
    def test_forward_with_ort_disabled(self):
        model = SimpleModel()
        args = OrtifyArgs(ort_enabled=False)
        wrapped = ortify(model, args)

        x = torch.randn(2, 10)
        output = wrapped(x)

        expected = model(x)
        assert torch.allclose(output, expected)

    def test_forward_with_ort_enabled(self):
        model = SimpleModel()
        model.eval()
        args = OrtifyArgs(ort_enabled=True)
        wrapped = ortify(model, args)

        x = torch.randn(2, 10)
        output = wrapped(x)

        with torch.no_grad():
            expected = model(x)

        assert output.shape == expected.shape
        assert torch.allclose(output, expected, atol=1e-5)

    def test_onnx_export_caching(self):
        model = SimpleModel()
        args = OrtifyArgs(ort_enabled=True)
        wrapped = ortify(model, args)

        x = torch.randn(2, 10)
        wrapped(x)
        first_session = wrapped._session

        wrapped(x)
        assert wrapped._session is first_session

    def test_custom_export_path(self):
        model = SimpleModel()
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "model.onnx"
            args = OrtifyArgs(ort_enabled=True, export_path=export_path)
            wrapped = ortify(model, args)

            x = torch.randn(2, 10)
            wrapped(x)

            assert export_path.exists()

    def test_output_shape_matches(self):
        model = SimpleModel()
        args = OrtifyArgs(ort_enabled=True)
        wrapped = ortify(model, args)

        x = torch.randn(4, 10)
        output = wrapped(x)

        assert output.shape == (4, 5)

    def test_bypasses_onnxruntime_args_to_inference_session(self, monkeypatch):
        captured: dict[str, object] = {}

        class DummyOutput:
            def __init__(self, name):
                self.name = name

        class DummySession:
            def __init__(self, path, **kwargs):
                captured["path"] = path
                captured["kwargs"] = kwargs

            def get_outputs(self):
                return [DummyOutput("output_0")]

            def run(self, output_names, inputs):
                return [inputs["input_0"]]

        def fake_export(*args, **kwargs):
            captured["export_kwargs"] = kwargs
            return None

        monkeypatch.setattr("ortify.core.torch.onnx.export", fake_export)
        monkeypatch.setattr("ortify.core.ort.InferenceSession", DummySession)

        model = SimpleModel()
        args = OrtifyArgs(
            onnxruntime_args={
                "providers": ["AzureExecutionProvider"],
                "sess_options": object(),
            }
        )
        wrapped = ortify(model, args)

        x = torch.randn(2, 10)
        output = wrapped(x)

        assert torch.equal(output, x)
        assert captured["kwargs"] == args.onnxruntime_args
        assert captured["export_kwargs"]["dynamo"] is False


class TestMultipleInputs:
    def test_multiple_tensor_inputs(self):
        class TwoInputModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 5)

            def forward(self, x, y):
                return self.linear(x + y)

        model = TwoInputModel()
        model.eval()
        args = OrtifyArgs(ort_enabled=True)
        wrapped = ortify(model, args)

        x = torch.randn(2, 10)
        y = torch.randn(2, 10)
        output = wrapped(x, y)

        with torch.no_grad():
            expected = model(x, y)

        assert torch.allclose(output, expected, atol=1e-5)
