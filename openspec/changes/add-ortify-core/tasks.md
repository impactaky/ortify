## 1. Project Setup
- [ ] 1.1 Create pyproject.toml with dependencies (torch, onnx, onnxruntime)
- [ ] 1.2 Create package structure `src/ortify/`

## 2. Core Implementation
- [ ] 2.1 Implement `OrtifyWrapper` class that wraps nn.Module
- [ ] 2.2 Implement ONNX export logic in wrapped forward
- [ ] 2.3 Implement ONNX Runtime inference session management
- [ ] 2.4 Implement `ortify()` function as main entry point

## 3. Testing
- [ ] 3.1 Add pytest and test dependencies
- [ ] 3.2 Write tests for basic module wrapping
- [ ] 3.3 Write tests for ONNX export and inference
