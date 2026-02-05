# Tests

## Running Tests

Compare ONNX runtime output against PyTorch ground truth:

```bash
uv run pytest tests/test_onnx_vs_pytorch.py
```

Or use the convenience script from the project root:

```bash
./test.sh
```

## Fixture Files

The fixture files contain ground truth NER (and classification) results from the native PyTorch GLiNER/GLiNER2 models. These are used to verify that the ONNX runtime produces matching results.

- `gliner1.fixtures.json` - NER results from GLiNER v2.1 models
- `gliner2.fixtures.json` - Classification + NER results from GLiNER2 models

Both files have already been generated and committed to the repository.

## Regenerating Fixtures

If you need to regenerate the fixtures (e.g., after updating test cases):

```bash
uv run python tests/generate_fixtures.py
```

This requires the native `gliner` and `gliner2` PyTorch packages to be installed, plus CUDA for GPU acceleration.

### Models Used

**GLiNER1** (NER only):
- `urchade/gliner_small-v2.1`
- `urchade/gliner_multi-v2.1`
- `urchade/gliner_large-v2.1`

**GLiNER2** (Classification + NER):
- `fastino/gliner2-multi-v1`
- `fastino/gliner2-base-v1`
- `fastino/gliner2-large-v1`
