"""
Custom exceptions for the Fruit Image Classification project.
"""


class FruitClassifierError(Exception):
    """
    Base exception for all project-specific errors.
    """


class ConfigurationError(FruitClassifierError):
    """
    Raised when configuration loading or validation fails.
    """


class DatasetError(FruitClassifierError):
    """
    Raised when dataset loading or validation fails.
    """


class ModelError(FruitClassifierError):
    """
    Raised when model creation or loading fails.
    """


class TrainingError(FruitClassifierError):
    """
    Raised when the training pipeline encounters an error.
    """


class InferenceError(FruitClassifierError):
    """
    Raised when inference fails.
    """
