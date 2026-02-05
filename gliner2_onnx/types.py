"""Type definitions for GLiNER2 ONNX runtime."""

from dataclasses import dataclass
from typing import TypedDict


class OnnxModelFiles(TypedDict):
    """ONNX model file paths for a single precision level."""

    encoder: str
    classifier: str
    span_rep: str
    count_embed: str


class GLiNER2Config(TypedDict):
    """GLiNER2 ONNX configuration schema."""

    max_width: int
    special_tokens: dict[str, int]
    onnx_files: dict[str, OnnxModelFiles]  # precision -> model files


@dataclass
class Entity:
    """Extracted named entity.

    Attributes:
        text: The entity text as it appears in the source
        label: The entity label/type
        start: Character offset where entity begins
        end: Character offset where entity ends
        score: Confidence score (0-1)
    """

    text: str
    label: str
    start: int
    end: int
    score: float
