# gliner2-onnx

GLiNER2 ONNX runtime for Python. Runs GLiNER2 models without PyTorch.

This library is experimental. The API may change between versions.

## Features

- Zero-shot NER and text classification
- Runs with ONNX Runtime (no PyTorch dependency)
- FP32 and FP16 precision support
- GPU acceleration via CUDA

All other GLiNER2 features such as JSON export are not supported. 

## Installation

```bash
pip install gliner2-onnx
```

## NER

```python
from gliner2_onnx import GLiNER2ONNXRuntime

runtime = GLiNER2ONNXRuntime.from_pretrained("lmoe/gliner2-large-v1-onnx")

entities = runtime.extract_entities(
    "John works at Google in Seattle",
    ["person", "organization", "location"]
)
# [
#   Entity(text='John', label='person', start=0, end=4, score=0.98),
#   Entity(text='Google', label='organization', start=14, end=20, score=0.97),
#   Entity(text='Seattle', label='location', start=24, end=31, score=0.96)
# ]
```

## Classification

```python
from gliner2_onnx import GLiNER2ONNXRuntime

runtime = GLiNER2ONNXRuntime.from_pretrained("lmoe/gliner2-large-v1-onnx")

# Single-label classification
result = runtime.classify(
    "Buy milk from the store",
    ["shopping", "work", "entertainment"]
)
# {'shopping': 0.95}

# Multi-label classification
result = runtime.classify(
    "Buy milk and finish the report",
    ["shopping", "work", "entertainment"],
    threshold=0.3,
    multi_label=True
)
# {'shopping': 0.85, 'work': 0.72}
```

## CUDA

To use CUDA for GPU acceleration:

```python
runtime = GLiNER2ONNXRuntime.from_pretrained(
    "lmoe/gliner2-large-v1-onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
```

## Precision

Both FP32 and FP16 models are supported. Only the requested precision is downloaded.

```python
runtime = GLiNER2ONNXRuntime.from_pretrained(
    "lmoe/gliner2-large-v1-onnx",
    precision="fp16"
)
```

## Models

Pre-exported ONNX models:

| Model | HuggingFace |
|-------|-------------|
| gliner2-large-v1 | [lmoe/gliner2-large-v1-onnx](https://huggingface.co/lmoe/gliner2-large-v1-onnx) |
| gliner2-multi-v1 | [lmoe/gliner2-multi-v1-onnx](https://huggingface.co/lmoe/gliner2-multi-v1-onnx) |

Note: `gliner2-base-v1` is not supported (uses a different architecture).

## Exporting Models

To export your own models, clone the repository and use make:

```bash
git clone https://github.com/lmoe/gliner2-onnx
cd gliner2-onnx

# FP32 only
make onnx-export MODEL=fastino/gliner2-large-v1

# FP32 + FP16
make onnx-export MODEL=fastino/gliner2-large-v1 QUANTIZE=fp16
```

Output is saved to `model_out/<model-name>/`.

## JavaScript/TypeScript

For Node.js, see [@lmoe/gliner-onnx.js](https://github.com/lmoe/gliner-onnx.js).

## Credits

- [fastino-ai/GLiNER2](https://github.com/fastino-ai/GLiNER2) - Original GLiNER2 implementation
- [fastino/gliner2-large-v1](https://huggingface.co/fastino/gliner2-large-v1) - Pre-trained models

## License

MIT
