# GLiNER2 ONNX Export Architecture

This document details the conversion of GLiNER2 from PyTorch to ONNX, including the challenges encountered and solutions implemented.

## Overview

GLiNER2 is a unified model for Named Entity Recognition (NER) and text classification. Converting it to ONNX enables:

- **Smaller deployments** - No need to ship PyTorch (~2GB) and its dependencies
- **Cross-platform inference** - Run in Node.js, Python, C++, Rust, etc.
- **Containerization** - Much smaller Docker images without the full ML stack
- **Edge deployment** - ONNX Runtime is optimized for various hardware

## Supported Models

The exporter supports GLiNER2 models using the CountLSTM architecture:

| Model | ONNX Size | Notes |
|-------|-----------|-------|
| `fastino/gliner2-multi-v1` | ~1.23 GB | Multilingual |
| `fastino/gliner2-large-v1` | ~1.95 GB | English, best accuracy |

**Not supported:** `fastino/gliner2-base-v1` uses CountLSTMv2 (different GRU architecture).

Quantized versions (FP16) reduce size by ~50% with minimal accuracy loss.

Output structure:
```
model_out/
├── gliner2-base-v1/
├── gliner2-large-v1/
```

**Key Challenge:** GLiNER2's architecture includes components that don't export cleanly to ONNX, requiring custom solutions.

---

## Model Architecture

### Components

```
GLiNER2 (486M params)
├── encoder (DeBERTa-v3-large)     433.9M params  → encoder.onnx (~1.65 GB)
├── span_rep (SpanMarkerV0)         29.4M params  → span_rep.onnx (~112 MB)
├── classifier (Sequential MLP)      2.1M params  → classifier.onnx (~8 MB)
├── count_embed (CountLSTM)         18.9M params  → count_embed_unrolled.onnx (~8 MB)
└── count_pred (Sequential MLP)      2.1M params  → NOT EXPORTED
```

### Why count_pred is not exported (simplification)

`count_pred` predicts how many entities of each type exist in the text. The native inference **does** use it:

```python
# Native GLiNER2 inference (engine.py:623-636)
count_logits = self.count_pred(embs[0].unsqueeze(0))
pred_count = int(count_logits.argmax(dim=1).item())
struct_proj = self.count_embed(embs[1:], pred_count)  # Uses predicted count
```

With `count > 1`, `count_embed` produces multiple "slots" per label, allowing the model to assign different spans to different instances of the same entity type.

**Our simplification:** We hardcode `count=1`, which:
- Produces a single transformed embedding per label
- Scores all spans against this single embedding
- Relies on threshold filtering to find multiple entities

This passes our test suite (6/6 NER) but may behave differently than native in edge cases with many entities of the same type. A more complete implementation would export `count_pred` and use dynamic counts.

### Data Flow

```
                    ┌─────────────────────────────────────────┐
                    │              CLASSIFICATION              │
                    │                                         │
Input: "Buy milk"   │  ( [P] category ( [L] shop [L] work ) ) [SEP_TEXT] buy milk
Labels: [shop,work] │            │                            │
                    │            ▼                            │
                    │       ┌─────────┐                       │
                    │       │ Encoder │                       │
                    │       └────┬────┘                       │
                    │            │                            │
                    │            ▼                            │
                    │  Extract embeddings at [L] positions    │
                    │            │                            │
                    │            ▼                            │
                    │     ┌────────────┐                      │
                    │     │ Classifier │ → sigmoid → probs    │
                    │     └────────────┘                      │
                    └─────────────────────────────────────────┘

                    ┌─────────────────────────────────────────┐
                    │                   NER                    │
                    │                                         │
Input: "John at Google"                                       │
Labels: [person, org] │  ( [P] entities ( [E] person [E] org ) ) [SEP_TEXT] john at google
                    │            │                            │
                    │            ▼                            │
                    │       ┌─────────┐                       │
                    │       │ Encoder │                       │
                    │       └────┬────┘                       │
                    │            │                            │
                    │     ┌──────┴──────┐                     │
                    │     ▼             ▼                     │
                    │  Text tokens   [E] positions            │
                    │     │             │                     │
                    │     ▼             ▼                     │
                    │ ┌─────────┐  ┌─────────────┐           │
                    │ │ SpanRep │  │ CountEmbed  │           │
                    │ └────┬────┘  └──────┬──────┘           │
                    │      │              │                   │
                    │      └──────┬───────┘                   │
                    │             ▼                           │
                    │   spans @ labels.T → sigmoid → scores   │
                    └─────────────────────────────────────────┘
```

---

## Input Formats

### Classification

GLiNER2 uses the same schema-based format as NER, with `[L]` tokens to mark labels:

```
( [P] category ( [L] <label1> [L] <label2> ... ) ) [SEP_TEXT] <text lowercase>
```

Example:
```
Input:  "Buy milk from the store"
Labels: ["shopping", "work", "entertainment"]

Tokenized:
▁( [P] ▁category ▁( [L] ▁shopping [L] ▁work [L] ▁entertainment ▁) ▁) [SEP_TEXT] ▁buy ▁milk ...

Token IDs:
[287, 128003, 3150, 287, 128007, 8428, 128007, 789, 128007, 5765, 1263, 1263, 128002, ...]
 ^(^  ^[P]^              ^[L]^          ^[L]^        ^[L]^                    ^[SEP_TEXT]^
```

The classifier extracts hidden states at `[L]` positions and passes them through an MLP.

### NER

GLiNER2 uses a structured schema format for NER:

```
( [P] entities ( [E] <label1> [E] <label2> ... ) ) [SEP_TEXT] <text lowercase>
```

Example:
```
Input:  "John works at Google"
Labels: ["person", "organization"]

Tokenized:
▁( [P] ▁entities ▁( [E] ▁person [E] ▁organization ▁) ▁) [SEP_TEXT] ▁john ▁works ▁at ▁google

Token IDs:
[287, 128003, 6967, 287, 128005, 1782, 128005, 1416, 1263, 1263, 128002, 1361, 1364, 311, 4826]
 ^(^  ^[P]^              ^[E]^          ^[E]^                    ^[SEP_TEXT]^ ^text tokens^
```

**Critical Notes:**
- Labels are marked with `[E]` tokens (not `[P]`)
- Text comes AFTER the schema, separated by `[SEP_TEXT]`
- Text is lowercased
- The nested parentheses are part of the schema structure

---

## Special Tokens

| Token | ID | Purpose |
|-------|-----|---------|
| `[CLS]` | 1 | Sequence start |
| `[SEP]` | 2 | Separator |
| `[SEP_STRUCT]` | 128001 | Schema structure separator |
| `[SEP_TEXT]` | 128002 | Schema/text separator |
| `[P]` | 128003 | Schema type marker ("entities", "category") |
| `[C]` | 128004 | (unused in this runtime) |
| `[E]` | 128005 | Entity label marker (NER) |
| `[R]` | 128006 | Relation marker |
| `[L]` | 128007 | Classification label marker |

---

## Export Challenges & Solutions

### Challenge 1: Encoder Export

**Problem:** DeBERTa-v3-large has complex attention patterns with relative position encodings.

**Solution:** Wrap the encoder to expose only `input_ids` and `attention_mask` inputs, returning `last_hidden_state`:

```python
class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
```

Export with dynamic axes for variable sequence lengths:
```python
torch.onnx.export(
    wrapper,
    (dummy_input_ids, dummy_attention_mask),
    "encoder.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["hidden_states"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "hidden_states": {0: "batch", 1: "seq"},
    },
    opset_version=17,
)
```

### Challenge 2: SpanRep Layer

**Problem:** The `SpanMarkerV0` layer has a complex forward pass that gathers embeddings at span start/end positions.

**Architecture:**
```
Input: hidden_states (batch, seq, 1024)

project_start: Linear(1024 → 4096) → ReLU → Dropout → Linear(4096 → 1024)
project_end:   Linear(1024 → 4096) → ReLU → Dropout → Linear(4096 → 1024)

1. Project ALL tokens through start/end MLPs
2. Gather at span start/end positions
3. CONCATENATE (not add!): [start_proj; end_proj] → (batch, num_spans, 2048)
4. ReLU
5. out_project: Linear(2048 → 4096) → ReLU → Dropout → Linear(4096 → 1024)

Output: (batch, num_spans, 1024)
```

**Solution:** Create a wrapper that explicitly implements the gather operations:

```python
class SpanRepCore(nn.Module):
    def forward(self, hidden_states, span_start_idx, span_end_idx):
        # Project all tokens
        start_rep = self.project_start(hidden_states)  # (batch, seq, hidden)
        end_rep = self.project_end(hidden_states)

        # Gather at span positions
        start_idx_expanded = span_start_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        end_idx_expanded = span_end_idx.unsqueeze(-1).expand(-1, -1, self.hidden_size)

        start_span_rep = torch.gather(start_rep, 1, start_idx_expanded)
        end_span_rep = torch.gather(end_rep, 1, end_idx_expanded)

        # Concatenate and project
        cat = torch.cat([start_span_rep, end_span_rep], dim=-1)  # (batch, spans, 2048)
        cat = torch.relu(cat)
        return self.out_project(cat)
```

### Challenge 3: CountEmbed (GRU) - The Hard One

**Problem:** The `CountLSTM` layer uses a GRU for sequential processing:

```python
class CountLSTM(nn.Module):
    def __init__(self, hidden_size, max_count=20):
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.projector = MLP(hidden_size * 2 → hidden_size)

    def forward(self, label_emb, count):
        # label_emb: (M, hidden) - embeddings at [E] positions
        # count: int - number of instances to extract

        pos_seq = self.pos_embedding(range(count))  # (count, hidden)
        pos_seq = pos_seq.expand(count, M, hidden)   # (count, M, hidden)

        h0 = label_emb.unsqueeze(0)  # (1, M, hidden)
        output, _ = self.gru(pos_seq, h0)  # Sequential processing!

        return self.projector(concat(output, label_emb))
```

**Why ONNX Export Fails:**
- The GRU processes `count` steps sequentially with hidden state
- `count` is dynamic (determined at runtime)
- ONNX's GRU operator has issues with the specific tensor shapes and dynamic slicing

**Error:**
```
ONNXRuntimeError: Non-zero status code returned while running Slice node.
Name:'node_Slice_15' Status Message: Starts must be a 1-D array
```

**Solution: Unroll the GRU for count=1**

For inference, we always use `count=1` (we want to find all entities, not a specific count). A single GRU step can be expressed as pure matrix operations:

```python
# GRU equations for ONE step:
z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)  # update gate
r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)  # reset gate
n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))  # new gate
h_new = (1 - z) * n + z * h

# For count=1:
#   x = pos_embedding[0]  (fixed constant - position 0)
#   h = label_embeddings  (input)
```

**Implementation:**

```python
class UnrolledCountEmbed(nn.Module):
    def __init__(self, count_embed, hidden_size):
        # Extract GRU weights (packed as [reset, update, new])
        gru = count_embed.gru
        W_ih = gru.weight_ih_l0.data  # (3*hidden, hidden)
        W_hh = gru.weight_hh_l0.data
        b_ih = gru.bias_ih_l0.data
        b_hh = gru.bias_hh_l0.data

        # Split into components
        self.W_ir = W_ih[:hidden_size, :]
        self.W_iz = W_ih[hidden_size:2*hidden_size, :]
        self.W_in = W_ih[2*hidden_size:, :]
        # ... same for W_hh, biases

        # Fixed position embedding for count=0
        self.pos_0 = count_embed.pos_embedding.weight[0]

        # Copy the projector
        self.projector = count_embed.projector

    def forward(self, label_embeddings):
        h = label_embeddings  # (M, hidden)
        x = self.pos_0        # (hidden,)

        # Single GRU step
        z = torch.sigmoid(x @ self.W_iz.t() + self.b_iz + h @ self.W_hz.t() + self.b_hz)
        r = torch.sigmoid(x @ self.W_ir.t() + self.b_ir + h @ self.W_hr.t() + self.b_hr)
        n = torch.tanh(x @ self.W_in.t() + self.b_in + r * (h @ self.W_hn.t() + self.b_hn))
        h_new = (1 - z) * n + z * h

        # Projector
        return self.projector(torch.cat([h_new, h], dim=-1))
```

**Verification:**
```
Max difference between original and unrolled: 1.43e-06
```

The unrolled version produces mathematically identical results.

---

## Scoring Mechanism

### Classification Scoring

Simple: extract hidden states at `[L]` positions, pass through MLP, sigmoid.

```python
l_embeddings = hidden_states[0, l_positions, :]  # (num_labels, hidden)
logits = classifier(l_embeddings)  # (num_labels, 1)
probs = sigmoid(logits)
```

### NER Scoring

More complex: requires transforming label embeddings before scoring.

```python
# 1. Get label embeddings at [E] positions
label_emb = hidden_states[0, e_positions, :]  # (num_labels, hidden)

# 2. Transform via count_embed (unrolled)
transformed = count_embed(label_emb)  # (num_labels, hidden)

# 3. Get span representations
span_rep = span_rep_layer(text_hidden, span_indices)  # (num_spans, hidden)

# 4. Score via dot product
scores = span_rep @ transformed.T  # (num_spans, num_labels)
probs = sigmoid(scores)
```

**Why count_embed is necessary:**
The label embeddings at `[E]` positions are not directly comparable to span representations. The count_embed layer projects them into a "scoring-compatible space". Without this transformation, simple dot product scoring produces incorrect results (scores ~1.0 for wrong entities).

---

## Deduplication

GLiNER2 can assign multiple labels to the same span (e.g., "Apple Store" as both organization AND location). The deduplication strategy:

1. Sort entities by score (descending)
2. For each entity, check for overlaps with already-kept entities **of the same label**
3. If no overlap with same-label entities, keep it

```python
def deduplicate(entities):
    entities = sorted(entities, key=lambda e: e.score, reverse=True)
    kept = []
    for entity in entities:
        overlaps = False
        for kept_entity in kept:
            if entity.label != kept_entity.label:
                continue  # Different labels can overlap
            if entity.start < kept_entity.end and entity.end > kept_entity.start:
                overlaps = True
                break
        if not overlaps:
            kept.append(entity)
    return kept
```

---

## Exported Files (HuggingFace Style)

Output follows HuggingFace model repository conventions:

```
repo/
├── config.json              # Base transformer config (architectures, hidden_size, vocab_size)
├── gliner2_config.json      # GLiNER2-specific (max_width, special_tokens, onnx_files)
├── added_tokens.json        # Tokenizer files at root level
├── special_tokens_map.json
├── tokenizer.json
├── tokenizer_config.json
└── onnx/                    # ONNX files in subdirectory
    ├── encoder.onnx         # ~1.65 GB - DeBERTa-v3-large encoder
    ├── encoder_fp16.onnx    # ~800 MB (optional)
    ├── encoder_int8.onnx    # ~400 MB (optional)
    ├── classifier.onnx      # ~8 MB - Classification MLP head
    ├── span_rep.onnx        # ~112 MB - Span representation layer
    └── count_embed.onnx     # ~8 MB - Unrolled GRU for NER scoring
```

| File | Size | Description |
|------|------|-------------|
| `onnx/encoder.onnx` | ~1.65 GB | DeBERTa-v3-large encoder |
| `onnx/classifier.onnx` | ~8 MB | Classification MLP head |
| `onnx/span_rep.onnx` | ~112 MB | Span representation layer |
| `onnx/count_embed.onnx` | ~72 MB | Unrolled GRU for NER scoring |

**Total (fp32): ~1.85 GB**

### Quantized Variants

Export with `--quantize fp16` to create quantized variants:

| Variant | Suffix | Size Reduction | Accuracy Impact |
|---------|--------|----------------|-----------------|
| FP16 | `_fp16` | ~50% | Minimal |

Each ONNX file can have quantized variants (e.g., `encoder.onnx`, `encoder_fp16.onnx`).

---

## Runtime Flow

### Loading

```python
runtime = GLiNER2ONNXRuntime('./model_out/gliner2-large-v1')
# Loads: encoder, classifier, span_rep, count_embed, tokenizer
```

### Classification

```python
result = runtime.classify("Buy milk from store", ["shopping", "work"])
# 1. Build input: ( [P] category ( [L] shopping [L] work ) ) [SEP_TEXT] buy milk from store
# 2. Run encoder → hidden_states
# 3. Extract [L] positions → classifier → sigmoid → probs
# Returns: {"shopping": 0.95}
```

### NER

```python
entities = runtime.extract_entities("John at Google", ["person", "organization"])
# 1. Build input: ( [P] entities ( [E] person [E] org ) ) [SEP_TEXT] john at google
# 2. Run encoder → hidden_states
# 3. Extract [E] positions → count_embed_unrolled → transformed labels
# 4. Extract text tokens → span_rep → span representations
# 5. Score: spans @ transformed.T → sigmoid → filter by threshold
# Returns: [Entity("John", "person", ...), Entity("Google", "organization", ...)]
```

---

## Performance

| Operation | ONNX | Native PyTorch | Speedup |
|-----------|------|----------------|---------|
| Classification | ~100-150ms | ~200-300ms | ~2x |
| NER | ~150-200ms | ~220-250ms | ~1.3x |

Note: Times are for CPU inference. GPU would be faster for both.

---

## Lessons Learned

1. **Don't assume input format**: GLiNER2's NER input format (`( [P] entities ( [E] ... ) ) [SEP_TEXT] text`) is significantly different from what you might expect from reading the paper.

2. **Trace the actual computation**: The only reliable way to understand the input format was to hook into the encoder and capture what the native model actually sends.

3. **GRUs can be unrolled**: If you only need a fixed number of steps (count=1), you can convert recurrent layers to pure matrix operations.

4. **Verify mathematically**: Always verify that your ONNX output matches the original PyTorch output (diff should be < 1e-5).

5. **Test against native**: The comprehensive test suite compares every output against native PyTorch to catch regressions.
