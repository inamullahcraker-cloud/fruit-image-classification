"""
Configuration loading utilities.

This module is responsible for loading project configuration
from YAML files. It serves as the central entry point for
configuration management throughout the application.
"""

from pathlib import Path
from typing import Any

import yaml

from src.config.schema import (
    Config,
    DatasetConfig,
    TrainingConfig,
    ModelConfig,
    LoggingConfig,
    ExperimentConfig,
    InferenceConfig,
)


def load_yaml(config_path: Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    ValueError
        If the YAML file is empty.

    yaml.YAMLError
        If the YAML file contains invalid syntax.
    """
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    return config


def build_config(config_dict: dict[str, Any]) -> Config:
    """
    Convert a configuration dictionary into a strongly typed
    Config object.
    """

    return Config(
        dataset=DatasetConfig(
            image_size=config_dict["dataset"]["image_size"],
            num_workers=config_dict["dataset"]["num_workers"],
            train_dir=config_dict["dataset"]["train_dir"],
            val_dir=config_dict["dataset"]["val_dir"],
            test_dir=config_dict["dataset"]["test_dir"],
        ),

        training=TrainingConfig(
            batch_size=config_dict["training"]["batch_size"],
            epochs=config_dict["training"]["epochs"],
            learning_rate=config_dict["training"]["learning_rate"],
            weight_decay=config_dict["training"]["weight_decay"],
            device=config_dict["training"]["device"],
            seed=config_dict["training"]["seed"],
            loss_name=config_dict["training"]["loss_name"],
            label_smoothing=config_dict["training"]["label_smoothing"],
            lr_step_size=config_dict["training"]["lr_step_size"],
            lr_gamma=config_dict["training"]["lr_gamma"],
            use_amp=config_dict["training"]["use_amp"],
        ),

        model=ModelConfig(
            name=config_dict["model"]["name"],
            num_classes=config_dict["model"]["num_classes"],
            pretrained=config_dict["model"]["pretrained"],
        ),

        logging=LoggingConfig(
            log_dir=config_dict["logging"]["log_dir"],
            log_level=config_dict["logging"]["log_level"],
        ),

        experiment=ExperimentConfig(
            experiment_name=config_dict["experiment"]["experiment_name"],
            output_dir=config_dict["experiment"]["output_dir"],
        ),

        inference=InferenceConfig(
            checkpoint_path=config_dict["inference"]["checkpoint_path"],
            class_mapping=config_dict["inference"]["class_mapping"],
            confidence_threshold=config_dict["inference"]["confidence_threshold"],
        ),
    )


def load_config(config_path: str | Path) -> Config:
    """
    Load the project configuration.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    Config
        Project configuration object.
    """
    config_dict = load_yaml(Path(config_path))
    return build_config(config_dict)