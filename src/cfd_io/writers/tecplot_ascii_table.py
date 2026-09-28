"""Tecplot ASCII writer for `Table`.

Writes a `Table` to a Tecplot ASCII (``.dat``) file as an ordered
zone with ``I=N, J=1, K=1, F=POINT``.  This is valid Tecplot format
readable by Tecplot 360, FieldView, and most post-processors.

Output format::

    TITLE = "cfd-io"
    VARIABLES = "eta", "fp", "g_cf", "tau"
    ZONE T="Zone 1", I=500, J=1, K=1, F=POINT
     0.00000000E+00  0.00000000E+00  0.00000000E+00  1.67614432E+00
    ...

Note: this writer handles `Table` only.  For structured CFD grid +
flow data use `write_tecplot_ascii` (which requires a `Dataset`).
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

from cfd_io.table import Table

# --------------------------------------------------
# set up logger
# --------------------------------------------------
logger = logging.getLogger(__name__)


# --------------------------------------------------
# public API
# --------------------------------------------------
def write_table_tecplot_ascii(
    fpath: str | Path,
    table: Table,
    *,
    title: str = "cfd-io",
    zone_title: str = "Zone 1",
) -> Path:
    """Write a `Table` to a Tecplot ASCII ``.dat`` file.

    The table is written as a single ordered zone with ``J=1, K=1``
    so it loads as a 1-D line in Tecplot.  Column names become
    Tecplot variable names in the order they appear in the table.

    The ``title`` and ``zone_title`` can be overridden from
    ``table.metadata`` via the keys ``"title"`` and ``"zone_title"``
    respectively; explicit keyword arguments take precedence.

    Args:
        fpath: Output file path.
        table: Table to write.
        title: Tecplot TITLE string.
        zone_title: Tecplot ZONE T string.

    Returns:
        Path to the created file.
    """
    # convert to Path object
    fpath = Path(fpath)

    # allow metadata to supply defaults, keyword args take precedence
    if title == "cfd-io" and "title" in table.metadata:
        title = str(table.metadata["title"])
    if zone_title == "Zone 1" and "zone_title" in table.metadata:
        zone_title = str(table.metadata["zone_title"])

    n_rows = table.n_rows
    n_cols = table.n_columns
    names = table.names

    # -- write file --------------------------------------------------------
    with open(fpath, "w") as fobj:

        # write TITLE line
        fobj.write(f'TITLE = "{title}"\n')

        # write VARIABLES line: quoted column names
        var_str = ", ".join(f'"{name}"' for name in names)
        fobj.write(f"VARIABLES = {var_str}\n")

        # write ZONE line: 1-D ordered zone
        fobj.write(f'ZONE T="{zone_title}", I={n_rows}, J=1, K=1, F=POINT\n')

        # write data rows: one row per table entry, all columns on one line
        arrays = [table.columns[name] for name in names]
        for row_idx in range(n_rows):
            vals = " ".join(f"{arrays[col_idx][row_idx]:16.8E}" for col_idx in range(n_cols))
            fobj.write(vals + "\n")

    logger.info(
        "wrote %s  (%d rows, %d vars: %s)",
        fpath, n_rows, n_cols, ", ".join(names),
    )

    return fpath
