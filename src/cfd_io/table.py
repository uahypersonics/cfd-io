"""Tabular scientific data with no topology.

Defines `Table` — a lightweight carrier for ordered columns of 1-D data.

Typical uses: line cuts, wall distributions, spectra, N-factor curves,
convergence histories, digitized reference data.

This is intentionally dumb: no interpolation, filtering, units logic,
or physics methods.  Storage + validation + I/O only.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# --------------------------------------------------
# Table dataclass
# --------------------------------------------------
@dataclass(slots=True)
class Table:
    """Tabular scientific data with no topology.

    ``columns`` is a dict mapping column name to a 1-D numpy array.
    All arrays must have the same length.  Column names must be
    non-empty strings; duplicates are impossible by construction
    (dict keys are unique).

    ``metadata`` is an optional dict for free-form annotations such
    as title, source reference, units, or solver conditions.

    Examples:
        >>> import numpy as np
        >>> from cfd_io.table import Table
        >>> t = Table(columns={"eta": np.linspace(0, 6, 50), "fp": np.ones(50)})
        >>> t.n_rows
        50
        >>> t.names
        ['eta', 'fp']
    """

    columns: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- validation -------------------------------------------------------
    def __post_init__(self) -> None:
        # validate at least one column
        if not self.columns:
            raise ValueError("Table must have at least one column")

        # validate column names are non-empty strings
        for name in self.columns:
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"column names must be non-empty strings, got {name!r}"
                )

        # validate all arrays are 1-D numpy arrays
        for name, arr in self.columns.items():
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"column {name!r} must be a numpy ndarray, "
                    f"got {type(arr).__name__}"
                )
            if arr.ndim != 1:
                raise ValueError(
                    f"column {name!r} must be 1-D, got shape {arr.shape}"
                )

        # validate all arrays have the same length
        lengths = {name: len(arr) for name, arr in self.columns.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(
                f"all columns must have the same length; got {lengths}"
            )

    # -- properties -------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """Ordered list of column names."""
        return list(self.columns.keys())

    @property
    def n_rows(self) -> int:
        """Number of rows (length of each column array)."""
        return len(next(iter(self.columns.values())))

    @property
    def n_columns(self) -> int:
        """Number of columns."""
        return len(self.columns)

    @property
    def shape(self) -> tuple[int, int]:
        """``(n_rows, n_columns)`` — mirrors numpy convention."""
        return (self.n_rows, self.n_columns)
