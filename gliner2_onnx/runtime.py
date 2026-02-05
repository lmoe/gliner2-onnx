"""GLiNER2 ONNX Runtime - NER and classification without PyTorch."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .constants import (
    CLASSIFICATION_TASK_NAME,
    CONFIG_FILE,
    GLINER2_CONFIG_FILE,
    NER_TASK_NAME,
    ONNX_ATTENTION_MASK,
    ONNX_HIDDEN_STATE,
    ONNX_HIDDEN_STATES,
    ONNX_INPUT_IDS,
    ONNX_LABEL_EMBEDDINGS,
    ONNX_SPAN_END_IDX,
    ONNX_SPAN_START_IDX,
    REQUIRED_SPECIAL_TOKENS,
    SCHEMA_CLOSE,
    SCHEMA_OPEN,
    TOKEN_E,
    TOKEN_L,
    TOKEN_P,
    TOKEN_SEP_TEXT,
    WORD_PATTERN,
    Precision,
)
from .exceptions import ConfigurationError, ModelNotFoundError
from .types import Entity, GLiNER2Config, OnnxModelFiles


def _validate_precision(onnx_files: dict[str, OnnxModelFiles], precision: str) -> OnnxModelFiles:
    available = list(onnx_files.keys())
    if not available:
        raise ConfigurationError(f"No onnx_files found in {GLINER2_CONFIG_FILE}")
    if precision not in available:
        raise ConfigurationError(f"Precision '{precision}' not available. Available: {available}")
    return onnx_files[precision]


class GLiNER2ONNXRuntime:
    """
    ONNX-based runtime for GLiNER2 classification and NER.

    Example:
        >>> runtime = GLiNER2ONNXRuntime.from_pretrained("lmoe/gliner2-large-v1-onnx")
        >>> entities = runtime.extract_entities("John works at Google", ["person", "org"])
        >>> result = runtime.classify("Buy milk", ["shopping", "work"])
    """

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        precision: Precision = "fp32",
        providers: list[str] | None = None,
        revision: str | None = None,
    ) -> "GLiNER2ONNXRuntime":
        """
        Load a GLiNER2 ONNX model from HuggingFace Hub.

        Args:
            model_id: HuggingFace model ID (e.g., "lmoe/gliner2-large-v1-onnx")
            precision: Model precision ("fp32" or "fp16").
                      Only downloads the requested precision variant.
                      Available precisions are defined in the model's config.
            providers: ONNX execution providers (e.g., ["CUDAExecutionProvider"]).
                      Defaults to ["CPUExecutionProvider"].
                      Use onnxruntime.get_available_providers() to see available options.
            revision: Model revision (branch, tag, or commit hash)
            cache_dir: Directory to cache downloaded models

        Returns:
            GLiNER2ONNXRuntime instance
        """
        from huggingface_hub import hf_hub_download, snapshot_download

        config_path = hf_hub_download(
            repo_id=model_id,
            filename=GLINER2_CONFIG_FILE,
            revision=revision,
        )

        with Path(config_path).open() as f:
            config: GLiNER2Config = json.load(f)

        onnx_files = _validate_precision(config.get("onnx_files", {}), precision)
        onnx_patterns: list[str] = [
            onnx_files["encoder"],
            onnx_files["classifier"],
            onnx_files["span_rep"],
            onnx_files["count_embed"],
        ]
        onnx_data_patterns: list[str] = [f"{p}.data" for p in onnx_patterns]

        allow_patterns: list[str] = [
            "*.json",
            *onnx_patterns,
            *onnx_data_patterns,
        ]

        model_path = snapshot_download(
            repo_id=model_id,
            revision=revision,
            allow_patterns=allow_patterns,
        )

        return cls(model_path, precision=precision, providers=providers)

    def __init__(
        self,
        model_path: str | Path,
        precision: Precision = "fp32",
        providers: list[str] | None = None,
    ):
        """
        Initialize GLiNER2 ONNX runtime from a local directory.

        Args:
            model_path: Directory containing ONNX models and config
            precision: Model precision ("fp32" or "fp16")
            providers: ONNX execution providers (e.g., ["CUDAExecutionProvider"]).
                      Defaults to ["CPUExecutionProvider"].
                      Use onnxruntime.get_available_providers() to see available options.

        Raises:
            ModelNotFoundError: If required model files are missing
            ConfigurationError: If config files are missing or invalid
        """
        self.model_path = Path(model_path)
        self._validate_model_path()

        self._validate_base_config()
        config = self._load_gliner2_config()

        self.max_width = config["max_width"]
        self.special_tokens = config["special_tokens"]
        self.precision = precision

        onnx_files = _validate_precision(config["onnx_files"], precision)

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self._load_onnx_models(onnx_files, providers)

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))

    def _validate_model_path(self) -> None:
        """Validate that model directory exists."""
        if not self.model_path.is_dir():
            raise ModelNotFoundError(f"Model directory not found: {self.model_path}")

    def _validate_base_config(self) -> None:
        """Validate that base config exists and is valid JSON."""
        config_path = self.model_path / CONFIG_FILE
        if not config_path.exists():
            raise ConfigurationError(f"{CONFIG_FILE} not found in {self.model_path}")

        try:
            with config_path.open() as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid {CONFIG_FILE}: {e}") from e

    def _load_gliner2_config(self) -> GLiNER2Config:
        """Load GLiNER2-specific config."""
        config_path = self.model_path / GLINER2_CONFIG_FILE
        if not config_path.exists():
            raise ConfigurationError(f"{GLINER2_CONFIG_FILE} not found in {self.model_path}")

        try:
            with config_path.open() as f:
                raw_config: dict[str, object] = json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid {GLINER2_CONFIG_FILE}: {e}") from e

        max_width = raw_config.get("max_width")
        if not isinstance(max_width, int):
            raise ConfigurationError(f"{GLINER2_CONFIG_FILE} missing or invalid max_width")

        special_tokens_raw = raw_config.get("special_tokens")
        if not isinstance(special_tokens_raw, dict):
            raise ConfigurationError(f"{GLINER2_CONFIG_FILE} missing or invalid special_tokens")

        try:
            special_tokens = {str(k): int(v) for k, v in special_tokens_raw.items()}
        except (TypeError, ValueError) as e:
            raise ConfigurationError(f"{GLINER2_CONFIG_FILE} special_tokens values must be integers: {e}") from e

        missing = [t for t in REQUIRED_SPECIAL_TOKENS if t not in special_tokens]
        if missing:
            raise ConfigurationError(f"{GLINER2_CONFIG_FILE} missing special tokens: {missing}")

        onnx_files_raw = raw_config.get("onnx_files")
        if not isinstance(onnx_files_raw, dict):
            raise ConfigurationError(f"{GLINER2_CONFIG_FILE} missing or invalid onnx_files")

        return GLiNER2Config(
            max_width=max_width,
            special_tokens=special_tokens,
            onnx_files=onnx_files_raw,
        )

    def _load_onnx_models(self, onnx_files: OnnxModelFiles, providers: list[str]) -> None:
        """Load all ONNX model files using paths from config."""
        self.encoder = self._load_model(self.model_path / onnx_files["encoder"], providers)
        self.classifier = self._load_model(self.model_path / onnx_files["classifier"], providers)
        self.span_rep = self._load_model(self.model_path / onnx_files["span_rep"], providers)
        self.count_embed = self._load_model(self.model_path / onnx_files["count_embed"], providers)

    def _load_model(self, path: Path, providers: list[str]) -> ort.InferenceSession:
        """Load a single ONNX model."""
        if not path.exists():
            raise ModelNotFoundError(f"Model not found: {path}")
        return ort.InferenceSession(str(path), providers=providers)

    def classify(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
        *,
        multi_label: bool = False,
    ) -> dict[str, float]:
        """
        Classify text into one or more categories.

        Args:
            text: Text to classify
            labels: Candidate labels
            threshold: Minimum score threshold (for multi_label mode)
            multi_label: If True, return all labels above threshold;
                        if False, return only the best label

        Returns:
            Dict mapping label(s) to score(s)
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if not labels:
            raise ValueError("Labels cannot be empty")

        input_ids, attention_mask, label_positions = self._build_classification_input(text, labels)
        hidden_states = self._encode(input_ids, attention_mask)
        label_embeddings = hidden_states[0, label_positions, :]
        logits = self.classifier.run(None, {ONNX_HIDDEN_STATE: label_embeddings})[0].flatten()

        if multi_label:
            probs = self._sigmoid(logits)
            results = {label: float(prob) for label, prob in zip(labels, probs, strict=True)}
            return {k: v for k, v in results.items() if v >= threshold}

        probs = self._softmax(logits)
        results = {label: float(prob) for label, prob in zip(labels, probs, strict=True)}
        best = max(results.keys(), key=lambda k: results[k])
        return {best: results[best]}

    def _build_schema_prefix(
        self,
        task_name: str,
        labels: list[str],
        label_token_key: str,
    ) -> tuple[list[int], list[int]]:
        """Build schema prefix tokens: ( [P] task ( [L/E] label1 [L/E] label2 ... ) ) [SEP_TEXT]"""
        p_id = self.special_tokens[TOKEN_P]
        label_token_id = self.special_tokens[label_token_key]
        sep_text_id = self.special_tokens[TOKEN_SEP_TEXT]

        tokens: list[int] = []
        tokens.extend(self.tokenizer.encode(SCHEMA_OPEN, add_special_tokens=False))
        tokens.append(p_id)
        tokens.extend(self.tokenizer.encode(task_name, add_special_tokens=False))
        tokens.extend(self.tokenizer.encode(SCHEMA_OPEN, add_special_tokens=False))

        label_positions = []
        for label in labels:
            label_positions.append(len(tokens))
            tokens.append(label_token_id)
            tokens.extend(self.tokenizer.encode(label, add_special_tokens=False))

        tokens.extend(self.tokenizer.encode(SCHEMA_CLOSE, add_special_tokens=False))
        tokens.extend(self.tokenizer.encode(SCHEMA_CLOSE, add_special_tokens=False))
        tokens.append(sep_text_id)

        return tokens, label_positions

    def _build_classification_input(
        self,
        text: str,
        labels: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Build input for classification task."""
        tokens, label_positions = self._build_schema_prefix(CLASSIFICATION_TASK_NAME, labels, TOKEN_L)

        for match in WORD_PATTERN.finditer(text.lower()):
            tokens.extend(self.tokenizer.encode(match.group(), add_special_tokens=False))

        input_ids = np.array([tokens], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        return input_ids, attention_mask, label_positions

    def extract_entities(
        self,
        text: str,
        labels: list[str],
        threshold: float = 0.5,
    ) -> list[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Text to analyze
            labels: Entity types to extract (e.g., ["person", "organization"])
            threshold: Minimum confidence score

        Returns:
            List of Entity objects with text, label, position, and score
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        if not labels:
            raise ValueError("Labels cannot be empty")

        input_ids, attention_mask, e_positions, word_offsets, text_start_idx, first_token_positions = self._build_ner_input(text, labels)

        hidden_states = self._encode(input_ids, attention_mask)
        label_embeddings = hidden_states[0, e_positions, :]
        text_hidden = hidden_states[0, text_start_idx:, :]

        num_words = len(word_offsets)
        if num_words == 0:
            return []

        word_span_start, word_span_end = self._generate_spans(num_words)

        token_span_start = np.array([first_token_positions[i] for i in word_span_start], dtype=np.int64)
        token_span_end = np.array([first_token_positions[i] for i in word_span_end], dtype=np.int64)

        span_rep = self._get_span_rep(
            text_hidden[np.newaxis, :, :],
            token_span_start[np.newaxis, :],
            token_span_end[np.newaxis, :],
        )[0]

        scores = self._compute_span_label_scores(span_rep, label_embeddings)

        entities = self._collect_entities(scores, word_span_start, word_span_end, word_offsets, labels, text, threshold)

        return self._deduplicate_entities(entities)

    def _build_ner_input(
        self,
        text: str,
        labels: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[int], list[tuple[int, int]], int, list[int]]:
        """Build input for NER task with word-level span support."""
        tokens, e_positions = self._build_schema_prefix(NER_TASK_NAME, labels, TOKEN_E)
        text_start_idx = len(tokens)

        word_offsets: list[tuple[int, int]] = []
        first_token_positions: list[int] = []
        token_idx = 0

        for match in WORD_PATTERN.finditer(text.lower()):
            word_offsets.append((match.start(), match.end()))
            first_token_positions.append(token_idx)

            word_tokens = self.tokenizer.encode(match.group(), add_special_tokens=False)
            tokens.extend(word_tokens)
            token_idx += len(word_tokens)

        input_ids = np.array([tokens], dtype=np.int64)
        attention_mask = np.ones_like(input_ids)

        return input_ids, attention_mask, e_positions, word_offsets, text_start_idx, first_token_positions

    def _generate_spans(self, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
        """Generate all valid span (start, end) pairs up to max_width."""
        start_indices = []
        end_indices = []

        for i in range(seq_len):
            for j in range(min(self.max_width, seq_len - i)):
                start_indices.append(i)
                end_indices.append(i + j)

        return np.array(start_indices, dtype=np.int64), np.array(end_indices, dtype=np.int64)

    def _collect_entities(
        self,
        scores: np.ndarray,
        word_span_start: np.ndarray,
        word_span_end: np.ndarray,
        word_offsets: list[tuple[int, int]],
        labels: list[str],
        text: str,
        threshold: float,
    ) -> list[Entity]:
        """Collect entities that exceed the threshold."""
        entities = []

        for span_idx in range(word_span_start.shape[0]):
            start_word = word_span_start[span_idx]
            end_word = word_span_end[span_idx]

            for label_idx, label in enumerate(labels):
                score = scores[span_idx, label_idx]
                if score >= threshold:
                    char_start = word_offsets[start_word][0]
                    char_end = word_offsets[end_word][1]
                    entities.append(
                        Entity(
                            text=text[char_start:char_end],
                            label=label,
                            start=char_start,
                            end=char_end,
                            score=float(score),
                        )
                    )

        return entities

    def _deduplicate_entities(self, entities: list[Entity]) -> list[Entity]:
        """Remove overlapping entities of the same label, keeping highest score."""
        if not entities:
            return []

        sorted_entities = sorted(entities, key=lambda e: e.score, reverse=True)
        kept: list[Entity] = []

        for entity in sorted_entities:
            overlaps = any(entity.label == kept_entity.label and entity.start < kept_entity.end and entity.end > kept_entity.start for kept_entity in kept)
            if not overlaps:
                kept.append(entity)

        return kept

    def _encode(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        """Run encoder on inputs."""
        result: np.ndarray = self.encoder.run(
            None,
            {ONNX_INPUT_IDS: input_ids, ONNX_ATTENTION_MASK: attention_mask},
        )[0]
        return result

    def _get_span_rep(
        self,
        hidden_states: np.ndarray,
        span_start: np.ndarray,
        span_end: np.ndarray,
    ) -> np.ndarray:
        """Get span representations from hidden states."""
        result: np.ndarray = self.span_rep.run(
            None,
            {
                ONNX_HIDDEN_STATES: hidden_states.astype(np.float32),
                ONNX_SPAN_START_IDX: span_start,
                ONNX_SPAN_END_IDX: span_end,
            },
        )[0]
        return result

    def _compute_span_label_scores(
        self,
        span_rep: np.ndarray,
        label_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Compute similarity scores between spans and labels."""
        transformed_labels = self.count_embed.run(
            None,
            {ONNX_LABEL_EMBEDDINGS: label_embeddings.astype(np.float32)},
        )[0]

        scores = np.einsum("sh,lh->sl", span_rep, transformed_labels)
        return self._sigmoid(scores)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        result = np.empty_like(x)
        pos_mask = x >= 0
        neg_mask = ~pos_mask
        result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
        exp_x = np.exp(x[neg_mask])
        result[neg_mask] = exp_x / (1 + exp_x)
        return result

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = float(np.max(x))
        exp_x = np.exp(x - x_max)
        return np.asarray(exp_x / np.sum(exp_x))
