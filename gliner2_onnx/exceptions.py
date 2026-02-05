"""Custom exceptions for GLiNER2 ONNX runtime."""


class GLiNER2Error(Exception):
    """Base exception for GLiNER2 ONNX runtime errors."""


class ModelNotFoundError(GLiNER2Error):
    """Raised when a required model file is not found."""


class ConfigurationError(GLiNER2Error):
    """Raised when configuration is invalid or missing."""
