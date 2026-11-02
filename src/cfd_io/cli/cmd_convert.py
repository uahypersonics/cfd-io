"""CLI handler for ``cfd-io convert``."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

import typer

from .helpers import load_attrs_from_file, parse_attr_pairs


# --------------------------------------------------
# subcommand: convert
# --------------------------------------------------
def cmd_convert(
    input_path: Path = typer.Argument(..., help="Input file path (flow file for split formats)."),
    output: Path = typer.Option(
        "output.h5",
        "--output", "-o",
        help="Output file path.",
    ),
    grid: Path | None = typer.Option(
        None,
        "--grid-in", "-g",
        help="Input grid file (required for split format input).",
    ),
    output_grid: Path | None = typer.Option(
        None,
        "--grid-out", "-G",
        help="Output grid file (required for split format output).",
    ),
    attr: list[str] | None = typer.Option(
        None, "--attr", "-a",
        help="Metadata as key=value (repeatable, e.g. --attr mach=6.0 --attr re=1e6).",
    ),
    attr_file: Path | None = typer.Option(
        None, "--attr-file", "-A",
        help="Load attributes from a file (.dat legacy or .json). "
             "Merged with --attr; --attr takes precedence on conflicts.",
    ),
) -> None:
    """Convert CFD data between formats (dispatched by file extension).

    Pure format conversion only. Use ``cfd-ops transpose`` to swap axes
    or other ``cfd-ops`` commands for any data transformation.

    \b
    Examples:
        # HDF5 -> Plot3D split (grid + flow)
        cfd-io convert base_flow.h5 -o flow.q -G grid.x

        # Plot3D split -> HDF5 (single file)
        cfd-io convert flow.q -g grid.x -o base_flow.h5

        # convert and stamp flow-condition attributes
        cfd-io convert flow.q -g grid.x -o base_flow.h5 \\
            --attr mach=6.0 --attr re1=1e7 --attr temp_inf=54.0

        # stamp all attributes from a legacy flow_conditions.dat
        cfd-io convert flow.q -g grid.x -o base_flow.h5 -A flow_conditions.dat

        # stamp from JSON, override one value on the command line
        cfd-io convert flow.q -g grid.x -o base_flow.h5 \\
            -A flow_conditions.json --attr mach=6.0
    """

    # lazy import of converter (deferred to keep CLI startup fast)
    from cfd_io.convert_mod import do_convert

    # load attributes from file first (lower precedence)
    attrs: dict[str, float | str] = {}
    if attr_file:
        attrs.update(load_attrs_from_file(attr_file))

    # parse --attr key=value pairs (these override any file-loaded attrs)
    if attr:
        attrs.update(parse_attr_pairs(attr))

    # run the conversion; report read/write problems as one line, not a traceback
    try:
        result = do_convert(
            input_path=input_path,
            output_path=output,
            input_grid=grid,
            output_grid=output_grid,
            attrs=attrs if attrs else None,
        )
    except (OSError, ValueError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from None

    typer.echo(f"wrote {result}")
