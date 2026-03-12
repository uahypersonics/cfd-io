"""Canonical CFD dataset types.

Defines the core data model for cfd-io:

- `StructuredGrid` -- structured (i, j, k) grid coordinates
- `UnstructuredGrid` -- unstructured node/cell mesh (stub)
- `Field` -- single flow variable with mesh association
- `Dataset` -- one grid + one flow snapshot + metadata

"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


# --------------------------------------------------
# structured grid dataclass
# --------------------------------------------------
@dataclass
class StructuredGrid:
    """Structured (i, j, k) grid coordinates.

    Each coordinate array has shape ``(ni, nj, nk)``.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray

    # post-init validation to ensure coordinate arrays have matching shapes
    def __post_init__(self) -> None:
        if self.x.shape != self.y.shape or self.x.shape != self.z.shape:
            raise ValueError(
                f"coordinate shape mismatch: x={self.x.shape}, "
                f"y={self.y.shape}, z={self.z.shape}"
            )

    # convenience property to get shape of coordiante arrays
    @property
    def shape(self) -> tuple[int, ...]:
        """Grid dimensions ``(ni, nj, nk)``."""
        return self.x.shape

    # convenience property to get number of active extents, i.e. is the dataset 1D, 2D, or 3D
    @property
    def ndim(self) -> int:
        """Number of active extents (dimensions with size > 1)."""
        return sum(1 for s in self.shape if s > 1)


# --------------------------------------------------
#  unstructured grid dataclass (stub)
# --------------------------------------------------
@dataclass
class UnstructuredGrid:
    """Unstructured node/cell mesh (stub - no readers produce this yet)."""

    points: np.ndarray  # (npts, ndim)
    connectivity: np.ndarray
    cell_types: np.ndarray


Grid = StructuredGrid | UnstructuredGrid

# --------------------------------------------------
# field type dataclass
# --------------------------------------------------
@dataclass
class Field:
    """Single flow variable array with mesh association metadata."""

    data: np.ndarray
    association: Literal["node", "cell", "face"] = "node"


# --------------------------------------------------
# dataset dataclass
# --------------------------------------------------
@dataclass
class Dataset:
    """One grid + one flow snapshot + metadata.

    This is the canonical data object for cfd-io.  Readers produce
    it; writers consume it; downstream packages (flow-props, etc.)
    operate on it.
    """

    grid: Grid
    flow: dict[str, Field] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
