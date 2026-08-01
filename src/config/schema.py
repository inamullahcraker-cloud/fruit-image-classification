"""
Configuration schema definitions.

This module defines the strongly typed configuration objects
used throughout the project.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class DatasetConfig:
    """
    Dataset configuration.
    """

    image_size: int
    num_workers: int
    train_dir: str
    val_dir: str
    test_dir: str


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float

    device: str
    seed: int

    loss_name: str
    label_smoothing: float

    lr_step_size: int
    lr_gamma: float

    use_amp: bool


@dataclass(slots=True)
class ModelConfig:
    """
    Model configuration.
    """

    name: str
    num_classes: int
    pretrained: bool



@dataclass(slots=True)
class InferenceConfig:
    """
    Inference configuration.
    """

    checkpoint_path: str
    class_mapping: str
    confidence_threshold: float
    #batch_size: int
@dataclass(slots=True)
class LoggingConfig:
    """
    Logging configuration.
    """

    log_dir: str
    log_level: str


@dataclass(slots=True)
class ExperimentConfig:
    """
    Experiment tracking configuration.
    """

    experiment_name: str
    output_dir: str


@dataclass(slots=True)
class Config:
    """
    Root project configuration.
    """

    dataset: DatasetConfig
    training: TrainingConfig
    model: ModelConfig
    logging: LoggingConfig
    experiment: ExperimentConfig
    inference: InferenceConfig
