## Context
This is the initial implementation of ortify, a library that wraps PyTorch modules to run on ONNX Runtime. The design needs to be simple and Pythonic while handling the complexity of ONNX export.

## Goals / Non-Goals
- Goals:
  - Provide simple `ortify(module, args)` API
  - Auto-export to ONNX on first forward pass
  - Cache ONNX session for subsequent calls
  - Support basic tensor inputs/outputs
- Non-Goals:
  - Dynamic shape support (initial version uses fixed shapes)
  - Multi-GPU/distributed support
  - Quantization/optimization (future enhancement)

## Decisions
- Decision: Use wrapper class pattern
  - The `ortify()` function returns an `OrtifyWrapper` instance that wraps the original module
  - Wrapper has a custom `forward()` that handles ONNX export and inference
  - This keeps the original module intact and allows fallback to PyTorch

- Decision: Lazy ONNX export on first forward call
  - Export happens when first input is available (we need input shapes)
  - ONNX file cached to temp directory or user-specified path
  - Session created once and reused

- Decision: Simple configuration via dataclass
  - `OrtifyArgs` dataclass holds configuration (ort_enabled, export_path, opset_version)
  - Clean separation between wrapping logic and configuration

## Risks / Trade-offs
- Risk: ONNX export may fail for some models → Document supported operations
- Trade-off: Fixed shapes simplify implementation but limit flexibility → Add dynamic shapes later

## Open Questions
- None for initial implementation
