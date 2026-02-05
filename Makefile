.PHONY: test lint fix build clean onnx-export

# Default model for ONNX export
MODEL ?= fastino/gliner2-large-v1
QUANTIZE ?=

test:
	uv run --extra test pytest tests/test_onnx_vs_pytorch.py

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy gliner2_onnx

fix:
	uv run ruff check --fix .
	uv run ruff format .

build:
	uv build

clean:
	rm -rf dist/ build/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Export GLiNER2 model to ONNX format (fp32 is always created)
# Usage:
#   make onnx-export                                    # fp32 only
#   make onnx-export MODEL=fastino/gliner2-multi-v1    # Custom model
#   make onnx-export QUANTIZE=fp16                     # fp32 + fp16
onnx-export:
	@echo "Exporting $(MODEL) to ONNX..."
	@MODEL_NAME=$$(basename "$(MODEL)") && \
	OUTPUT_DIR="./model_out/$$MODEL_NAME" && \
	echo "Output: $$OUTPUT_DIR" && \
	uv run --extra export python tools/export_model.py \
		--model "$(MODEL)" \
		--save-path "$$OUTPUT_DIR" \
		$(if $(QUANTIZE),--quantize $(QUANTIZE)) && \
	uv run --extra export python tools/export_count_embed.py \
		--model "$(MODEL)" \
		--save-path "$$OUTPUT_DIR" \
		$(if $(QUANTIZE),--quantize $(QUANTIZE)) && \
	echo "Export complete: $$OUTPUT_DIR"
