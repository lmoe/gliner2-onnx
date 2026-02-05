import re
from typing import Final, Literal

CONFIG_FILE: Final = "config.json"
GLINER2_CONFIG_FILE: Final = "gliner2_config.json"

Precision = Literal["fp32", "fp16"]

TOKEN_P: Final = "[P]"  # noqa: S105
TOKEN_L: Final = "[L]"  # noqa: S105
TOKEN_E: Final = "[E]"  # noqa: S105
TOKEN_SEP_TEXT: Final = "[SEP_TEXT]"  # noqa: S105

REQUIRED_SPECIAL_TOKENS: Final = (TOKEN_P, TOKEN_L, TOKEN_E, TOKEN_SEP_TEXT)

SCHEMA_OPEN: Final = "("
SCHEMA_CLOSE: Final = ")"
NER_TASK_NAME: Final = "entities"
CLASSIFICATION_TASK_NAME: Final = "category"

ONNX_INPUT_IDS: Final = "input_ids"
ONNX_ATTENTION_MASK: Final = "attention_mask"
ONNX_HIDDEN_STATE: Final = "hidden_state"
ONNX_HIDDEN_STATES: Final = "hidden_states"
ONNX_SPAN_START_IDX: Final = "span_start_idx"
ONNX_SPAN_END_IDX: Final = "span_end_idx"
ONNX_LABEL_EMBEDDINGS: Final = "label_embeddings"

# Regex pattern matching GLiNER2's WhitespaceTokenSplitter
# Matches: URLs, emails, @mentions, words (with hyphens/underscores), single non-whitespace chars
WORD_PATTERN: Final = re.compile(
    r"""(?:https?://[^\s]+|www\.[^\s]+)
    |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
    |@[a-z0-9_]+
    |\w+(?:[-_]\w+)*
    |\S""",
    re.VERBOSE | re.IGNORECASE,
)
