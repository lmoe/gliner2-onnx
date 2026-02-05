#!/usr/bin/env python3
"""
Export GLiNER2 to ONNX format.

Exports encoder, classifier, and span_rep components.
The count_embed component is exported separately by export_count_embed.py.

Usage:
  uv run python exporter/export_model.py --model fastino/gliner2-large-v1
  uv run python exporter/export_model.py --model fastino/gliner2-large-v1 --quantize q8
"""

import argparse
import json
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

warnings.filterwarnings("ignore")


class EncoderWrapper(nn.Module):
    """Wrapper for the DeBERTa encoder."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state


class ClassifierWrapper(nn.Module):
    """Wrapper for the classifier head."""

    def __init__(self, classifier: nn.Module):
        super().__init__()
        self.classifier = classifier

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden_states)


class SpanRepWrapper(nn.Module):
    """
    Wrapper for the span representation layer.

    The span_rep layer expects:
    - hidden_states: (batch, seq_len, hidden_size)
    - It internally generates spans based on max_width

    Output: (batch, seq_len, max_width, hidden_size)
    """

    def __init__(self, span_rep: nn.Module, max_width: int):
        super().__init__()
        self.span_rep = span_rep
        self.max_width = max_width

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        device = hidden_states.device

        # Generate all possible span indices
        spans_idx = []
        for i in range(seq_len):
            for j in range(self.max_width):
                if i + j < seq_len:
                    spans_idx.append((i, i + j))
                else:
                    spans_idx.append((0, 0))  # Padding with valid indices

        spans_idx_tensor = torch.tensor([spans_idx], dtype=torch.long, device=device)
        spans_idx_tensor = spans_idx_tensor.expand(batch_size, -1, -1)

        # Get span representations
        span_rep = self.span_rep(hidden_states, spans_idx_tensor)

        # Reshape to (batch, seq_len, max_width, hidden)
        return span_rep.view(batch_size, seq_len, self.max_width, -1)


class CountEmbedWrapper(nn.Module):
    """
    Wrapper for the count embedding layer (used in NER scoring).

    Takes schema embeddings and count, produces projected embeddings for scoring.
    """

    def __init__(self, count_embed: nn.Module):
        super().__init__()
        self.count_embed = count_embed

    def forward(self, schema_embeddings: torch.Tensor, count: torch.Tensor) -> torch.Tensor:
        """
        Args:
            schema_embeddings: (num_labels, hidden_size)
            count: scalar tensor with predicted count
        Returns:
            projected: (count, num_labels, hidden_size)
        """
        return self.count_embed(schema_embeddings, count.item())


class NERScorerWrapper(nn.Module):
    """
    Combined NER scoring: span_rep -> einsum with label embeddings.

    This is a simplified version that takes pre-computed components.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(
        self,
        span_rep: torch.Tensor,  # (batch, num_spans, hidden)
        label_embeddings: torch.Tensor,  # (num_labels, hidden)
    ) -> torch.Tensor:
        """
        Compute span-label similarity scores.

        Returns: (batch, num_spans, num_labels)
        """
        # Normalize for cosine similarity
        span_norm = span_rep / (span_rep.norm(dim=-1, keepdim=True) + 1e-8)
        label_norm = label_embeddings / (label_embeddings.norm(dim=-1, keepdim=True) + 1e-8)

        # Compute similarity
        return torch.einsum("bsh,lh->bsl", span_norm, label_norm)


def _get_file_size_mb(path: Path) -> float:
    """Get file size including external data file if present."""
    size = path.stat().st_size / (1024 * 1024)
    data_file = path.with_suffix(".onnx.data")
    if data_file.exists():
        size += data_file.stat().st_size / (1024 * 1024)
    return size


def export_encoder(model: Any, onnx_dir: Path, opset: int = 18) -> Path | None:
    """Export the encoder to ONNX."""
    import onnx

    print("\n" + "=" * 60)
    print("EXPORTING: ENCODER")
    print("=" * 60)

    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    batch_size, seq_len = 1, 128
    dummy_inputs = (
        torch.randint(0, 30000, (batch_size, seq_len)),
        torch.ones(batch_size, seq_len, dtype=torch.long),
    )

    onnx_path = onnx_dir / "encoder.onnx"
    print(f"Exporting to: {onnx_path}")
    print(f"  Hidden size: {model.hidden_size}")

    try:
        torch.onnx.export(
            wrapper,
            dummy_inputs,
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "attention_mask": {0: "batch", 1: "seq_len"},
                "hidden_states": {0: "batch", 1: "seq_len"},
            },
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

        # Verify
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"✗ Export failed: {e}")
        traceback.print_exc()
        return None
    else:
        size_mb = _get_file_size_mb(onnx_path)
        print(f"✓ Encoder exported ({size_mb:.1f} MB)")
        return onnx_path


def export_classifier(model: Any, onnx_dir: Path, opset: int = 18) -> Path | None:
    """Export the classifier head to ONNX."""
    import onnx

    print("\n" + "=" * 60)
    print("EXPORTING: CLASSIFIER HEAD")
    print("=" * 60)

    wrapper = ClassifierWrapper(model.classifier)
    wrapper.eval()

    hidden_size = model.hidden_size
    dummy_input = torch.randn(1, hidden_size)

    onnx_path = onnx_dir / "classifier.onnx"
    print(f"Exporting to: {onnx_path}")

    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_path),
            input_names=["hidden_state"],
            output_names=["logit"],
            dynamic_axes={"hidden_state": {0: "batch"}, "logit": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"✗ Export failed: {e}")
        traceback.print_exc()
        return None
    else:
        size_kb = onnx_path.stat().st_size / 1024
        print(f"✓ Classifier exported ({size_kb:.1f} KB)")
        return onnx_path


def export_span_rep(model: Any, onnx_dir: Path, opset: int = 18) -> Path | None:
    """Export the span representation layer to ONNX."""
    import onnx

    print("\n" + "=" * 60)
    print("EXPORTING: SPAN REPRESENTATION LAYER")
    print("=" * 60)

    span_rep = model.span_rep.span_rep_layer
    hidden_size = model.hidden_size
    max_width = model.max_width

    # The SpanMarkerV0 forward:
    # 1. Project ALL tokens through start/end MLPs
    # 2. Gather projected values at span start/end positions
    # 3. Concatenate (not add!) and ReLU
    # 4. Project through out_project

    class SpanRepCore(nn.Module):
        def __init__(self, span_layer: nn.Module, hidden_dim: int):
            super().__init__()
            self.project_start = span_layer.project_start
            self.project_end = span_layer.project_end
            self.out_project = span_layer.out_project
            self.hidden_size = hidden_dim

        def forward(
            self,
            hidden_states: torch.Tensor,  # (batch, seq_len, hidden)
            span_start_idx: torch.Tensor,  # (batch, num_spans)
            span_end_idx: torch.Tensor,  # (batch, num_spans)
        ) -> torch.Tensor:
            """Compute span representations from start/end indices."""
            # Step 1: Project ALL tokens through start/end MLPs
            start_rep = self.project_start(hidden_states)  # (batch, seq, hidden)
            end_rep = self.project_end(hidden_states)  # (batch, seq, hidden)

            # Step 2: Gather at span positions
            start_idx_expanded = span_start_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)
            end_idx_expanded = span_end_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)

            start_span_rep = torch.gather(start_rep, 1, start_idx_expanded)
            end_span_rep = torch.gather(end_rep, 1, end_idx_expanded)

            # Step 3: Concatenate and ReLU
            cat = torch.cat([start_span_rep, end_span_rep], dim=-1)
            cat = torch.relu(cat)

            # Step 4: Final projection
            return self.out_project(cat)

    wrapper = SpanRepCore(span_rep, hidden_size)
    wrapper.eval()

    # Dummy inputs - use proper span indices
    batch_size, seq_len = 1, 64

    dummy_hidden = torch.randn(batch_size, seq_len, hidden_size)

    # Generate proper span indices (matching GLiNER2's pattern)
    start_indices = []
    end_indices = []
    for i in range(seq_len):
        for j in range(max_width):
            if i + j < seq_len:
                start_indices.append(i)
                end_indices.append(i + j)
            else:
                start_indices.append(0)  # Padding
                end_indices.append(0)

    dummy_start = torch.tensor([start_indices], dtype=torch.long)
    dummy_end = torch.tensor([end_indices], dtype=torch.long)

    onnx_path = onnx_dir / "span_rep.onnx"
    print(f"Exporting to: {onnx_path}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Max width: {max_width}")

    try:
        torch.onnx.export(
            wrapper,
            (dummy_hidden, dummy_start, dummy_end),
            str(onnx_path),
            input_names=["hidden_states", "span_start_idx", "span_end_idx"],
            output_names=["span_representations"],
            dynamic_axes={
                "hidden_states": {0: "batch", 1: "seq_len"},
                "span_start_idx": {0: "batch", 1: "num_spans"},
                "span_end_idx": {0: "batch", 1: "num_spans"},
                "span_representations": {0: "batch", 1: "num_spans"},
            },
            opset_version=opset,
            dynamo=False,
        )

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"✗ Export failed: {e}")
        traceback.print_exc()
        return None
    else:
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"✓ Span rep exported ({size_mb:.1f} MB)")
        return onnx_path


def save_tokenizer_and_config(model: Any, save_path: Path, precisions: list[str]) -> None:
    """
    Save tokenizer and config files following HuggingFace conventions.

    Args:
        model: The GLiNER2 model
        save_path: Output directory
        precisions: List of precisions exported (e.g., ["fp32", "fp16"])

    Output structure:
      save_path/
      ├── config.json           # Base transformer config
      ├── gliner2_config.json   # GLiNER2-specific config
      ├── added_tokens.json     # Tokenizer files at root
      ├── special_tokens_map.json
      ├── tokenizer.json
      └── tokenizer_config.json
    """
    print("\n" + "=" * 60)
    print("SAVING: TOKENIZER & CONFIG (HuggingFace style)")
    print("=" * 60)

    tokenizer = model.processor.tokenizer

    # Save tokenizer files directly to root (HuggingFace convention)
    tokenizer.save_pretrained(str(save_path))
    print(f"✓ Tokenizer files saved to {save_path}")

    # Base config - only fields our runtime needs
    # NOTE: We intentionally omit "model_type" and "architectures" because
    # HuggingFace's AutoConfig doesn't recognize "gliner2" and throws warnings
    base_config = {
        "hidden_size": model.hidden_size,
        "vocab_size": tokenizer.vocab_size,
    }

    config_path = save_path / "config.json"
    with config_path.open("w") as f:
        json.dump(base_config, f, indent=2)
    print(f"✓ Base config saved to {config_path}")

    # Build onnx_files mapping for each precision
    onnx_files: dict[str, dict[str, str]] = {}
    for precision in precisions:
        suffix = "" if precision == "fp32" else f"_{precision}"
        onnx_files[precision] = {
            "encoder": f"onnx/encoder{suffix}.onnx",
            "classifier": f"onnx/classifier{suffix}.onnx",
            "span_rep": f"onnx/span_rep{suffix}.onnx",
            "count_embed": f"onnx/count_embed{suffix}.onnx",
        }

    # GLiNER2-specific config
    gliner2_config = {
        "max_width": model.max_width,
        "special_tokens": {
            "[SEP_STRUCT]": tokenizer.convert_tokens_to_ids("[SEP_STRUCT]"),
            "[SEP_TEXT]": tokenizer.convert_tokens_to_ids("[SEP_TEXT]"),
            "[P]": tokenizer.convert_tokens_to_ids("[P]"),
            "[C]": tokenizer.convert_tokens_to_ids("[C]"),
            "[E]": tokenizer.convert_tokens_to_ids("[E]"),
            "[R]": tokenizer.convert_tokens_to_ids("[R]"),
            "[L]": tokenizer.convert_tokens_to_ids("[L]"),
            "[EXAMPLE]": tokenizer.convert_tokens_to_ids("[EXAMPLE]"),
            "[OUTPUT]": tokenizer.convert_tokens_to_ids("[OUTPUT]"),
            "[DESCRIPTION]": tokenizer.convert_tokens_to_ids("[DESCRIPTION]"),
        },
        "onnx_files": onnx_files,
    }

    gliner2_config_path = save_path / "gliner2_config.json"
    with gliner2_config_path.open("w") as f:
        json.dump(gliner2_config, f, indent=2)
    print(f"✓ GLiNER2 config saved to {gliner2_config_path}")
    print(f"  Precisions: {precisions}")


def generate_readme(save_path: Path, model_name: str) -> None:
    """Generate README.md for HuggingFace model repo by combining root README with HF metadata."""
    model_short = model_name.rsplit("/", maxsplit=1)[-1]

    # Read the root README.md
    root_readme = Path(__file__).parent.parent / "README.md"
    readme_content = root_readme.read_text() if root_readme.exists() else f"# {model_short}-onnx\n\nGLiNER2 ONNX model.\n"

    # HuggingFace frontmatter
    frontmatter = f"""---
library_name: gliner2-onnx
base_model: {model_name}
tags:
  - onnx
  - gliner
  - gliner2
  - ner
  - named-entity-recognition
  - zero-shot
  - classification
license: mit
---

> **Experimental ONNX build** - Unofficial ONNX export of [{model_name}](https://huggingface.co/{model_name}).

"""

    readme_path = save_path / "README.md"
    with readme_path.open("w") as f:
        f.write(frontmatter + readme_content)
    print(f"✓ README.md saved to {readme_path}")


def test_exports(onnx_dir: Path, model: Any, *, classification_only: bool) -> None:
    """Test the exported ONNX models."""
    import onnxruntime as ort

    print("\n" + "=" * 60)
    print("TESTING EXPORTS")
    print("=" * 60)

    # Test encoder
    encoder_path = onnx_dir / "encoder.onnx"
    if encoder_path.exists():
        print("\nTesting encoder...")
        session = ort.InferenceSession(str(encoder_path), providers=["CPUExecutionProvider"])

        tokenizer = model.processor.tokenizer
        text = "John works at Google in San Francisco"
        inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)

        outputs = session.run(
            None,
            {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            },
        )
        print(f"  Input shape: {inputs['input_ids'].shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print("  ✓ Encoder works!")

    classifier_path = onnx_dir / "classifier.onnx"
    if classifier_path.exists():
        print("\nTesting classifier...")
        session = ort.InferenceSession(str(classifier_path), providers=["CPUExecutionProvider"])

        dummy = np.random.randn(1, model.hidden_size).astype(np.float32)
        outputs = session.run(None, {"hidden_state": dummy})
        print(f"  Input shape: {dummy.shape}")
        print(f"  Output shape: {outputs[0].shape}")
        print("  ✓ Classifier works!")

    if not classification_only:
        span_rep_path = onnx_dir / "span_rep.onnx"
        if span_rep_path.exists():
            print("\nTesting span_rep...")
            session = ort.InferenceSession(str(span_rep_path), providers=["CPUExecutionProvider"])

            batch, seq_len, num_spans = 1, 20, 50
            dummy_hidden = np.random.randn(batch, seq_len, model.hidden_size).astype(np.float32)
            dummy_start = np.random.randint(0, seq_len, (batch, num_spans)).astype(np.int64)
            dummy_end = np.maximum(
                dummy_start,
                np.random.randint(0, seq_len, (batch, num_spans)),
            ).astype(np.int64)

            outputs = session.run(
                None,
                {
                    "hidden_states": dummy_hidden,
                    "span_start_idx": dummy_start,
                    "span_end_idx": dummy_end,
                },
            )
            print(f"  Hidden shape: {dummy_hidden.shape}")
            print(f"  Span indices: {num_spans} spans")
            print(f"  Output shape: {outputs[0].shape}")
            print("  ✓ Span rep works!")


def get_model_output_path(model_name: str) -> Path:
    """Convert model name to output path: ./model_out/<safe_name>/"""
    safe_name = model_name.rsplit("/", maxsplit=1)[-1]
    return Path("./model_out") / safe_name


def quantize_model(onnx_path: Path, quantize_type: str) -> Path | None:
    """
    Create a quantized variant of an ONNX model.

    Following HuggingFace convention, quantized files are named:
      model.onnx -> model_fp16.onnx

    Returns the path to the new quantized file, or None if failed.
    """
    if quantize_type != "fp16":
        return None

    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    stem = onnx_path.stem
    parent = onnx_path.parent

    quantized_path = parent / f"{stem}_fp16.onnx"

    try:
        print(f"  Converting to FP16: {quantized_path.name}")

        onnx_model = onnx.load(str(onnx_path), load_external_data=True)
        model_fp16 = convert_float_to_float16(onnx_model, keep_io_types=True)

        original_data_file = onnx_path.with_suffix(".onnx.data")
        if original_data_file.exists():
            onnx.save_model(
                model_fp16,
                str(quantized_path),
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                location=f"{stem}_fp16.onnx.data",
                convert_attribute=False,
            )
        else:
            onnx.save(model_fp16, str(quantized_path))
    except Exception as e:
        print(f"  ⚠ Failed to convert {onnx_path.name} to FP16: {e}")
        return None
    else:
        return quantized_path


def verify_quantized_model(onnx_path: Path) -> bool:
    """Verify that a quantized ONNX model can be loaded by ONNX Runtime."""
    import onnxruntime as ort

    try:
        ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  ⚠ Failed to load {onnx_path.name}: {e}")
        return False
    else:
        return True


def create_quantized_variants(onnx_dir: Path, variants: list[str]) -> dict[str, list[Path]]:
    """
    Create quantized variants for all ONNX models in the directory.

    Args:
        onnx_dir: Directory containing .onnx files
        variants: List of quantization types (e.g., ["fp16", "int8"])

    Returns:
        Dict mapping model name to list of created variant paths
    """
    results: dict[str, list[Path]] = {}

    base_models = [f for f in onnx_dir.glob("*.onnx") if not any(f.stem.endswith(f"_{v}") for v in ["fp16", "int8"])]

    for onnx_path in base_models:
        model_name = onnx_path.stem
        results[model_name] = []

        for variant in variants:
            quantized = quantize_model(onnx_path, variant)
            if quantized and verify_quantized_model(quantized):
                results[model_name].append(quantized)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Export GLiNER2 to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="fastino/gliner2-large-v1",
        help="Model name (default: fastino/gliner2-large-v1)",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Output directory (default: ./model_out/<model_name>)",
    )
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument(
        "--quantize",
        type=str,
        nargs="*",
        default=[],
        help="Quantization variants to create: fp16 (e.g., --quantize fp16)",
    )

    args = parser.parse_args()

    valid_quantize = {"fp16"}
    for q in args.quantize:
        if q not in valid_quantize:
            print(f"Error: Invalid quantize option '{q}'. Valid options: {valid_quantize}")
            return 1

    save_path = Path(args.save_path) if args.save_path else get_model_output_path(args.model)
    save_path.mkdir(parents=True, exist_ok=True)

    onnx_dir = save_path / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GLiNER2 ONNX EXPORT (HuggingFace style)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Quantize variants: {args.quantize or ['none']}")
    print(f"Output: {save_path}")
    print(f"ONNX dir: {onnx_dir}")

    print("\n" + "=" * 60)
    print("LOADING MODEL")
    print("=" * 60)

    from gliner2 import GLiNER2

    model = GLiNER2.from_pretrained(args.model)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.1f}M")

    results: dict[str, Path | None] = {}

    results["encoder"] = export_encoder(model, onnx_dir, args.opset)
    results["classifier"] = export_classifier(model, onnx_dir, args.opset)
    results["span_rep"] = export_span_rep(model, onnx_dir, args.opset)

    # Note: count_embed is exported separately by export_count_embed.py

    precisions = ["fp32"]
    if args.quantize:
        print("\n" + "=" * 60)
        print("CREATING QUANTIZED VARIANTS")
        print("=" * 60)
        variants = create_quantized_variants(onnx_dir, args.quantize)
        for paths in variants.values():
            for p in paths:
                print(f"  ✓ {p.name}")
        precisions.extend(args.quantize)

    save_tokenizer_and_config(model, save_path, precisions)
    generate_readme(save_path, args.model)
    test_exports(onnx_dir, model, classification_only=False)

    # Summary
    print("\n" + "=" * 60)
    print("EXPORT SUMMARY")
    print("=" * 60)

    for name, path in results.items():
        status = "✓" if path else "✗"
        print(f"  {status} {name}: {path or 'FAILED'}")

    if all(results.values()):
        print("\n✓ ALL EXPORTS SUCCESSFUL")

        # Calculate total size (include onnx/ files)
        total_size = 0
        for f in save_path.rglob("*"):
            if f.is_file():
                total_size += f.stat().st_size

        print(f"\nTotal export size: {total_size / (1024 * 1024):.1f} MB")
        print(f"Output directory: {save_path}")

        print("\nExported files:")
        for f in sorted(save_path.rglob("*")):
            if f.is_file():
                rel_path = f.relative_to(save_path)
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {rel_path} ({size_mb:.2f} MB)")
    else:
        print("\n✗ SOME EXPORTS FAILED")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
