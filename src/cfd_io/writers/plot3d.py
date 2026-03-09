"""Plot3D structured grid writer -- dispatcher.

Delegates to the binary or ASCII writer based on the *binary* flag.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------

# write a grid dict to a Plot3D .x file (binary or ASCII)
def write_plot3d(
    fpath: str | Path,
    grid: dict[str, np.ndarray],
    flow: dict[str, np.ndarray] | None = None,
    attrs: dict[str, Any] | None = None,
    *,
    binary: bool = True,
) -> Path:
    """Write a grid dict to a Plot3D ``.x`` file.

    Args:
        fpath: Output file path.
        grid: Grid arrays ``{"x": (nx, ny, nz), "y": ..., "z": ...}``.
            If ``"z"`` is absent the file is written as 2-D.
        flow: Ignored -- Plot3D grid files carry no flow data.
        attrs: Ignored.
        binary: If ``True`` (default) write Fortran unformatted binary.
            If ``False`` write ASCII.

    Returns:
        Path to the created file.
    """
    # convert to Path object and validate inputs
    fpath = Path(fpath)

    if not grid:
        raise ValueError("grid dict must not be empty")
    if "x" not in grid or "y" not in grid:
        raise ValueError("grid must contain at least 'x' and 'y'")

    if grid["x"].ndim != 3:
        raise ValueError(f"expected 3-D arrays, got ndim={grid['x'].ndim}")

    # delegate to binary or ASCII writer based on flag
    if binary:
        from cfd_io.writers.plot3d_grid_binary import write_plot3d_grid_binary

        write_plot3d_grid_binary(fpath, grid)
    else:
        from cfd_io.writers.plot3d_grid_ascii import write_plot3d_grid_ascii

        write_plot3d_grid_ascii(fpath, grid)

    if flow:
        logger.debug("note: Plot3D grid format has no flow container; flow ignored")

    ni, nj, nk = grid["x"].shape
    is_3d = "z" in grid
    dim_str = f"{ni}x{nj}x{nk}" if is_3d else f"{ni}x{nj}"
    fmt_str = "binary" if binary else "ASCII"
    logger.info("wrote %s  (%s, %s)", fpath, dim_str, fmt_str)

    return fpath
