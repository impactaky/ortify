# Change: Add core ortify module

## Why
The project needs a core wrapper module that allows users to transparently run PyTorch models on ONNX Runtime. This provides the primary user-facing API for the library.

## What Changes
- Add `ortify()` function that wraps a PyTorch `nn.Module`
- Wrapped module automatically exports to ONNX and runs on ONNX Runtime when enabled
- Users get a drop-in replacement for their original module

## Impact
- Affected specs: `ortify` (new capability)
- Affected code: New module `src/ortify/`
