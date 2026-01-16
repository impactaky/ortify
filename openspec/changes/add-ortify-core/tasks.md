## 1. Project Setup
- [x] 1.1 Create pyproject.toml with dependencies (torch, onnx, onnxruntime, onnxscript)
- [x] 1.2 Create package structure `src/ortify/`

## 2. Core Implementation
- [x] 2.1 Implement `OrtifyWrapper` class that wraps nn.Module
- [x] 2.2 Implement ONNX export logic in wrapped forward
- [x] 2.3 Implement ONNX Runtime inference session management
- [x] 2.4 Implement `ortify()` function as main entry point

## 3. Testing
- [x] 3.1 Add pytest and test dependencies
- [x] 3.2 Write tests for basic module wrapping
- [x] 3.3 Write tests for ONNX export and inference
