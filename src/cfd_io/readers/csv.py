"""CSV reader for `Table`.

Reads a CSV (or whitespace-delimited) text file with a header row
into a `Table`.  No pandas dependency — uses `numpy.genfromtxt`.

Expected format (comma or whitespace separated)::

    eta,fp,g_cf,tau
    0.000000,0.000000,0.000000,1.676144
    0.024096,0.012578,0.012578,1.675981
    ...

Comment lines starting with ``#`` are skipped.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from cfd_io.table import Table

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------
def read_table_csv(
    fpath: str | Path,
    *,
    delimiter: str | None = None,
    comments: str = "#",
    encoding: str = "utf-8",
) -> Table:
    """Read a CSV or whitespace-delimited file into a `Table`.

    The first non-comment line is treated as the header and provides
    column names.  All subsequent lines are numeric data rows.

    Args:
        fpath: Path to the CSV file.
        delimiter: Column separator.  ``None`` means any whitespace
            (same as ``numpy.genfromtxt`` default).  Use ``","`` for
            standard CSV.
        comments: Lines beginning with this string are skipped.
        encoding: File encoding.

    Returns:
        `Table` with one column per header field.

    Raises:
        ValueError: If the file has no header, no data rows, or
            column names are not unique.
        FileNotFoundError: If ``fpath`` does not exist.
    """
    # convert to Path object
    fpath = Path(fpath)

    # read raw lines, strip comments and blanks to find the header
    with open(fpath, encoding=encoding) as fobj:
        raw_lines = fobj.readlines()

    # -- find header line (first non-comment, non-blank line) --------------
    header_line: str | None = None
    header_line_index: int = 0
    for idx, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped or stripped.startswith(comments):
            continue
        header_line = stripped
        header_line_index = idx
        break

    if header_line is None:
        raise ValueError(f"no header line found in {fpath}")

    # -- parse column names from header ------------------------------------
    # detect delimiter: if comma appears in header use it, else whitespace
    if delimiter is None:
        sep = "," if "," in header_line else None
    else:
        sep = delimiter

    if sep is not None:
        names = [n.strip().strip('"').strip("'") for n in header_line.split(sep)]
    else:
        names = header_line.split()

    # check for duplicate column names
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate column names in header: {names}")

    # -- read numeric data using genfromtxt --------------------------------
    # skip lines up to and including the header row
    skip_rows = header_line_index + 1

    data = np.genfromtxt(
        fpath,
        delimiter=sep,
        skip_header=skip_rows,
        comments=comments,
        encoding=encoding,
    )

    # genfromtxt returns 1-D array for a single data row; promote to 2-D
    if data.ndim == 1:
        data = data[np.newaxis, :]

    if data.shape[0] == 0:
        raise ValueError(f"no data rows found in {fpath}")

    if data.shape[1] != len(names):
        raise ValueError(
            f"header has {len(names)} columns but data has {data.shape[1]} columns"
        )

    # build columns dict
    columns: dict[str, np.ndarray] = {
        name: data[:, col_idx] for col_idx, name in enumerate(names)
    }

    logger.info(
        "read %s  (%d rows, %d columns: %s)",
        fpath, data.shape[0], len(names), ", ".join(names),
    )

    return Table(columns=columns)
