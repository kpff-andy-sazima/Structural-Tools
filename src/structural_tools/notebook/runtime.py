"""
Runtime helpers for structural_tools notebook execution.

This module provides execution context for calculation notebooks.

The goal is to hide implementation details such as environment variables,
nbconvert, and export scripts from calculation files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None) -> bool:
    """
    Convert a string value to a boolean.

    Parameters
    ----------
    value
        String representation of a boolean value.

    Returns
    -------
    bool
        True if the value represents an enabled state.
    """
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class Runtime:
    """
    Runtime information for the current notebook execution.

    Args:
        identifier: Unique calculation identifier.
        path: Path to the notebook being executed.
        generate_new_images: Whether image generation should run.
        nbconvert: Whether execution is occurring through nbconvert.
    """

    identifier: str
    path: Path | None
    generate_new_images: bool
    nbconvert: bool

    @property
    def filename(self) -> str | None:
        """
        Return the notebook filename.

        Returns
        -------
        str | None
            Notebook filename including extension.
        """
        return None if self.path is None else self.path.name


def image_path(
    self,
    name: str,
    extension: str = ".png",
) -> str:
    """
    Generate a calculation-specific image path.

    Args:
        name: Descriptive image name.
        extension: File extension including the leading dot.

    Returns:
        Relative image path.
    """


runtime = Runtime(
    identifier=os.getenv("CALC_ID", "unknown_calc"),
    path=(Path(path) if (path := os.getenv("CALC_PATH")) else None),
    generate_new_images=_as_bool(os.getenv("GENERATE_NEW_IMAGES")),
    nbconvert=_as_bool(os.getenv("NBCONVERT")),
)


def image_name(
    name: str,
    extension: str = ".png",
) -> str:
    """
    Convenience wrapper around runtime.image_path().

    Parameters
    ----------
    name
        Descriptive image name.

    extension
        File extension including the leading dot.

    Returns
    -------
    str
        Relative image path.
    """
    return runtime.image_path(
        name=name,
        extension=extension,
    )
