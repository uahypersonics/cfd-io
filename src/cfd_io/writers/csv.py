"""CSV writer for `Table`.

Writes a `Table` to a comma-separated text file with a header row.
Round-trips cleanly with `read_table_csv`.

Output format::

    eta,fp,g_cf,tau
    0.000000000000000000e+00,0.000000000000000000e+00,...
    ...
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from cfd_io.table import Table

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------
def write_table_csv(
    fpath: str | Path,
    table: Table,
    *,
    delimiter: str = ",",
    fmt: str = "%.18e",
    encoding: str = "utf-8",
) -> Path:
    """Write a `Table` to a CSV file.

    Args:
        fpath: Output file path.
        table: Table to write.
        delimiter: Column separator.  Defaults to ``","`` (standard CSV).
        fmt: ``numpy.savetxt`` format string for numeric values.
        encoding: File encoding.

    Returns:
        Path to the created file.
    """
    # convert to Path object
    fpath = Path(fpath)

    # build data matrix: (n_rows, n_columns)
    data = np.column_stack([table.columns[name] for name in table.names])

    # write header row and numeric data
    header = delimiter.join(table.names)
    np.savetxt(
        fpath,
        data,
        delimiter=delimiter,
        header=header,
        fmt=fmt,
        comments="",
        encoding=encoding,
    )

    logger.info(
        "wrote %s  (%d rows, %d columns: %s)",
        fpath, table.n_rows, table.n_columns, ", ".join(table.names),
    )

    return fpath
