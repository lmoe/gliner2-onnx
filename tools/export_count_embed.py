#!/usr/bin/env python3
"""
Export GLiNER2's count_embed layer with GRU unrolled for count=1.

The CountLSTM layer uses a GRU which doesn't export cleanly to ONNX.
For inference we always use count=1, so we can unroll the single GRU step
into explicit matrix operations that ARE ONNX-compatible.

GRU single-step equations:
    z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)  # update gate
    r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)  # reset gate
    n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))  # new gate
    h_new = (1 - z) * n + z * h

For count=1:
    x = pos_embedding[0] (fixed constant)
    h = label_embeddings (input)
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn


class UnrolledCountEmbed(nn.Module):
    """ONNX-exportable count_embed that unrolls the GRU for count=1."""

    def __init__(self, count_embed: nn.Module, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        # Extract GRU weights
        gru = count_embed.gru

        # GRU weights are packed as [W_ir, W_iz, W_in] for input weights
        # and [W_hr, W_hz, W_hn] for hidden weights
        # Each section is (hidden_size, input_size) or (hidden_size, hidden_size)

        # Weight matrices - shape: (3 * hidden_size, input_size)
        w_ih = gru.weight_ih_l0.data  # input weights: reset, update, new
        w_hh = gru.weight_hh_l0.data  # hidden weights: reset, update, new

        # Biases - shape: (3 * hidden_size,)
        b_ih = gru.bias_ih_l0.data
        b_hh = gru.bias_hh_l0.data

        # Split into r, z, n components
        self.W_ir = nn.Parameter(w_ih[:hidden_size, :], requires_grad=False)
        self.W_iz = nn.Parameter(w_ih[hidden_size : 2 * hidden_size, :], requires_grad=False)
        self.W_in = nn.Parameter(w_ih[2 * hidden_size :, :], requires_grad=False)

        self.W_hr = nn.Parameter(w_hh[:hidden_size, :], requires_grad=False)
        self.W_hz = nn.Parameter(w_hh[hidden_size : 2 * hidden_size, :], requires_grad=False)
        self.W_hn = nn.Parameter(w_hh[2 * hidden_size :, :], requires_grad=False)

        self.b_ir = nn.Parameter(b_ih[:hidden_size], requires_grad=False)
        self.b_iz = nn.Parameter(b_ih[hidden_size : 2 * hidden_size], requires_grad=False)
        self.b_in = nn.Parameter(b_ih[2 * hidden_size :], requires_grad=False)

        self.b_hr = nn.Parameter(b_hh[:hidden_size], requires_grad=False)
        self.b_hz = nn.Parameter(b_hh[hidden_size : 2 * hidden_size], requires_grad=False)
        self.b_hn = nn.Parameter(b_hh[2 * hidden_size :], requires_grad=False)

        # Position embedding for count=0 (fixed input)
        pos_0 = count_embed.pos_embedding.weight[0].data  # (hidden_size,)
        self.pos_0 = nn.Parameter(pos_0, requires_grad=False)

        # Copy the projector (MLP: Linear -> ReLU -> Linear)
        self.projector = count_embed.projector

    def forward(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Transform label embeddings using unrolled GRU.

        Args:
            label_embeddings: (num_labels, hidden_size) - embeddings at [P] positions

        Returns:
            (num_labels, hidden_size) - transformed embeddings for scoring
        """
        # h = label_embeddings, x = pos_0 (broadcast over labels)
        h = label_embeddings  # (M, D)
        x = self.pos_0  # (D,)

        # Compute GRU gates for all labels at once (x broadcasts over M labels)
        # Update gate: z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
        z = torch.sigmoid(torch.matmul(x, self.W_iz.t()) + self.b_iz + torch.matmul(h, self.W_hz.t()) + self.b_hz)

        # Reset gate: r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
        r = torch.sigmoid(torch.matmul(x, self.W_ir.t()) + self.b_ir + torch.matmul(h, self.W_hr.t()) + self.b_hr)

        # New gate: n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
        n = torch.tanh(torch.matmul(x, self.W_in.t()) + self.b_in + r * (torch.matmul(h, self.W_hn.t()) + self.b_hn))

        # Combine gates to compute new hidden state
        h_new = (1 - z) * n + z * h  # (M, D)

        # Projector expects concatenation of GRU output and original embeddings
        # Original shape was (gold_count_val, M, hidden_size) for both
        # For count=1, we have (1, M, D), squeeze to (M, D)
        combined = torch.cat([h_new, h], dim=-1)  # (M, 2D)

        return self.projector(combined)  # (M, D)


def verify_unrolled(
    count_embed: nn.Module,
    unrolled: nn.Module,
    num_labels: int = 5,
    hidden_size: int = 1024,
) -> bool:
    """Verify that unrolled version matches original."""
    label_emb = torch.randn(num_labels, hidden_size)

    # Original with count=1
    count_embed.eval()
    with torch.no_grad():
        orig_output = count_embed(label_emb, 1)  # (1, M, D)
        orig_output = orig_output.squeeze(0)  # (M, D)

    # Unrolled
    unrolled.eval()
    with torch.no_grad():
        unrolled_output = unrolled(label_emb)  # (M, D)

    # Compare
    diff = (orig_output - unrolled_output).abs().max().item()
    print(f"Max difference: {diff:.2e}")
    return diff < 1e-5


def get_model_output_path(model_name: str) -> Path:
    """Convert model name to output path: ./model_out/<safe_name>/"""
    safe_name = model_name.rsplit("/", maxsplit=1)[-1]
    return Path("./model_out") / safe_name


def export_count_embed_unrolled(model_name: str, save_path: Path, opset: int = 18) -> Path | None:
    """Export unrolled count_embed to ONNX."""
    import onnxruntime as ort
    from gliner2 import GLiNER2

    print("Loading GLiNER2 model...")
    model = GLiNER2.from_pretrained(model_name)
    model.eval()

    hidden_size = model.hidden_size

    print(f"\nCreating unrolled count_embed (hidden_size={hidden_size})...")
    unrolled = UnrolledCountEmbed(model.count_embed, hidden_size)
    unrolled.eval()

    print("\nVerifying unrolled matches original...")
    if not verify_unrolled(model.count_embed, unrolled, hidden_size=hidden_size):
        print("WARNING: Unrolled version doesn't match original!")
        return None

    print("✓ Verification passed!")

    # Export to ONNX in onnx/ subdirectory (HuggingFace convention)
    onnx_dir = save_path / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "count_embed.onnx"

    print(f"\nExporting to {onnx_path}...")

    # Dummy input
    num_labels = 10
    dummy_input = torch.randn(num_labels, hidden_size)

    torch.onnx.export(
        unrolled,
        dummy_input,
        str(onnx_path),
        input_names=["label_embeddings"],
        output_names=["transformed_embeddings"],
        dynamic_axes={
            "label_embeddings": {0: "num_labels"},
            "transformed_embeddings": {0: "num_labels"},
        },
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )

    print(f"✓ Exported to {onnx_path}")

    # Test ONNX runtime
    print("\nTesting ONNX runtime...")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    test_input = torch.randn(5, hidden_size)
    with torch.no_grad():
        expected = unrolled(test_input).numpy()

    onnx_output = session.run(None, {"label_embeddings": test_input.numpy()})[0]

    diff = np.abs(expected - onnx_output).max()
    print(f"ONNX vs PyTorch max diff: {diff:.2e}")

    if diff < 1e-4:
        print("✓ ONNX runtime verification passed!")
    else:
        print("⚠ ONNX output differs from PyTorch")

    return onnx_path


def quantize_count_embed(onnx_path: Path, variants: list[str]) -> list[Path]:
    """Create quantized variants of count_embed (fp16 only)."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    created: list[Path] = []
    stem = onnx_path.stem
    parent = onnx_path.parent

    for variant in variants:
        if variant == "fp16":
            try:
                quantized_path = parent / f"{stem}_fp16.onnx"
                print(f"  Converting to FP16: {quantized_path.name}")

                # Load model (includes external data if present)
                onnx_model = onnx.load(str(onnx_path), load_external_data=True)
                model_fp16 = convert_float_to_float16(onnx_model, keep_io_types=True)

                # Check if original had external data (large model)
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

                created.append(quantized_path)
            except Exception as e:
                print(f"  ⚠ Failed to convert to FP16: {e}")

    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Export GLiNER2 count_embed (unrolled GRU) to ONNX")
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
    parser.add_argument(
        "--quantize",
        type=str,
        nargs="*",
        default=[],
        help="Quantization variants to create: fp16",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)",
    )

    args = parser.parse_args()

    save_path = Path(args.save_path) if args.save_path else get_model_output_path(args.model)

    print("=" * 60)
    print("GLiNER2 COUNT_EMBED EXPORT (Unrolled GRU)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Output: {save_path}")
    print(f"Opset: {args.opset}")
    print(f"Quantize: {args.quantize or ['none']}")

    onnx_path = export_count_embed_unrolled(args.model, save_path, args.opset)

    if onnx_path and args.quantize:
        print("\nCreating quantized variants...")
        quantize_count_embed(onnx_path, args.quantize)


if __name__ == "__main__":
    main()
