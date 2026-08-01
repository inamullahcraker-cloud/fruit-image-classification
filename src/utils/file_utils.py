"""
File utility functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def file_exists(path: str | Path) -> bool:
    """Check whether a file exists."""
    return Path(path).is_file()


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write data to a JSON file."""
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file."""
    with Path(path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    """Write data to a YAML file."""
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            default_flow_style=False,
        )


def list_files(
    directory: str | Path,
    suffix: str | None = None,
) -> list[Path]:
    """
    List files inside a directory.

    Parameters
    ----------
    directory : str | Path
    suffix : str | None
        Example: ".jpg", ".png"
    """
    directory = Path(directory)

    if suffix:
        return sorted(directory.glob(f"*{suffix}"))

    return sorted(directory.iterdir())
