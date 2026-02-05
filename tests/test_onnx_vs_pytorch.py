#!/usr/bin/env python3
"""Test GLiNER2 ONNX runtime against pre-generated fixtures."""

import json
import sys
from pathlib import Path
from typing import TypedDict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gliner2_onnx import GLiNER2ONNXRuntime

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_PATH = Path(__file__).parent / "gliner2.fixtures.json"
SCORE_TOLERANCE = 0.05


class ClassificationFixture(TypedDict):
    text: str
    labels: list[str]
    expected_label: str
    expected_score: float


class EntityFixture(TypedDict):
    text: str
    label: str


class NERFixture(TypedDict):
    text: str
    labels: list[str]
    threshold: float
    expected: list[EntityFixture]


class ModelFixtures(TypedDict):
    classification: list[ClassificationFixture]
    ner: list[NERFixture]


def load_fixtures() -> dict[str, ModelFixtures]:
    """Load fixtures from JSON file."""
    if not FIXTURES_PATH.exists():
        pytest.skip(f"Fixtures not found at {FIXTURES_PATH}. Run 'uv run python tests/generate_fixtures.py' first.")
    with FIXTURES_PATH.open() as f:
        return json.load(f)


def get_available_model_precisions() -> list[tuple[str, str]]:
    """Get list of (model_key, precision) tuples that are available for testing."""
    if not FIXTURES_PATH.exists():
        return []

    with FIXTURES_PATH.open() as f:
        fixtures: dict[str, ModelFixtures] = json.load(f)

    available = []
    for model_key in fixtures:
        model_path = PROJECT_ROOT / "model_out" / model_key
        config_path = model_path / "gliner2_config.json"

        if not config_path.exists():
            continue

        with config_path.open() as f:
            config = json.load(f)

        precisions = list(config.get("onnx_files", {}).keys())
        available.extend((model_key, p) for p in precisions)

    return available


available_model_precisions = get_available_model_precisions()
if not available_model_precisions:
    pytest.skip("No exported models found. Run export first.", allow_module_level=True)


@pytest.fixture(scope="module")
def fixtures() -> dict[str, ModelFixtures]:
    """Load all fixtures."""
    return load_fixtures()


@pytest.fixture(scope="module", params=available_model_precisions, ids=lambda x: f"{x[0]}-{x[1]}")
def model_setup(request: pytest.FixtureRequest, fixtures: dict[str, ModelFixtures]) -> tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]:
    """Setup runtime for each model and precision combination."""
    model_key, precision = request.param
    model_path = PROJECT_ROOT / "model_out" / model_key
    runtime = GLiNER2ONNXRuntime(str(model_path), precision=precision, providers=["CPUExecutionProvider"])
    return model_key, precision, runtime, fixtures[model_key]


class TestClassification:
    """Classification tests."""

    def test_classification(self, model_setup: tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]) -> None:
        """Test classification predictions match expected results."""
        model_key, precision, runtime, model_fixtures = model_setup

        for fixture in model_fixtures["classification"]:
            text = fixture["text"]
            labels = fixture["labels"]
            expected_label = fixture["expected_label"]
            expected_score = fixture["expected_score"]

            result = runtime.classify(text, labels)
            actual_label = next(iter(result.keys()))
            actual_score = result[actual_label]

            assert actual_label == expected_label, (
                f"[{model_key}/{precision}] Label mismatch for '{text[:50]}...'\nExpected: {expected_label}, Got: {actual_label}"
            )
            assert abs(actual_score - expected_score) <= SCORE_TOLERANCE, (
                f"[{model_key}/{precision}] Score mismatch for '{text[:50]}...'\nExpected: {expected_score:.4f}, Got: {actual_score:.4f}"
            )


class TestNER:
    """NER tests."""

    def test_ner(self, model_setup: tuple[str, str, GLiNER2ONNXRuntime, ModelFixtures]) -> None:
        """Test NER extraction matches expected entities."""
        model_key, precision, runtime, model_fixtures = model_setup

        for fixture in model_fixtures["ner"]:
            text = fixture["text"]
            labels = fixture["labels"]
            threshold = fixture["threshold"]
            expected = fixture["expected"]

            entities = runtime.extract_entities(text, labels, threshold=threshold)

            actual_set = {(e.text, e.label) for e in entities}
            expected_set = {(e["text"], e["label"]) for e in expected}

            assert actual_set == expected_set, (
                f"[{model_key}/{precision}] Entity mismatch for '{text[:50]}...'\nMissing: {expected_set - actual_set}\nExtra: {actual_set - expected_set}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
