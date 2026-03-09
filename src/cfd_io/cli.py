"""cfd-io CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import typer

# --------------------------------------------------
# cli setup using Typer
# --------------------------------------------------
app = typer.Typer(
    name="cfd-io",
    help="CFD I/O toolkit: read, write, and convert between formats.",
    no_args_is_help=True,
    add_completion=False,
)


# --------------------------------------------------
# helper: configure console logging
# --------------------------------------------------
def _configure_logging(debug: bool) -> None:
    """Set up console logging."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)-8s %(message)s",
    )


# --------------------------------------------------
# subcommand: convert
# --------------------------------------------------
@app.command()
def convert(
    input_path: Path = typer.Argument(..., help="Input file path (flow file for split formats)."),
    output: Path = typer.Option(
        "output.h5",
        "--output", "-o",
        help="Output file path.",
    ),
    grid: Path | None = typer.Option(
        None,
        "--grid", "-g",
        help="Input grid file (required for split formats).",
    ),
    output_grid: Path | None = typer.Option(
        None,
        "--output-grid",
        help="Output grid file (required when writing split formats).",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
    mach: float | None = typer.Option(None, help="Store Mach number as metadata."),
    re: float | None = typer.Option(None, help="Store unit Reynolds number as metadata."),
    temp_inf: float | None = typer.Option(
        None, help="Store freestream temperature as metadata."
    ),
) -> None:
    """Convert CFD data between formats (dispatched by file extension)."""

    # set up logging
    _configure_logging(debug)

    # lazy import of converter (deferred to keep CLI startup fast)
    from cfd_io.convert_mod import do_convert

    # build optional attributes dict from CLI flags
    attrs = {}
    if mach is not None:
        attrs["mach"] = mach
    if re is not None:
        attrs["re1"] = re
    if temp_inf is not None:
        attrs["temp_inf"] = temp_inf

    # run the conversion
    result = do_convert(
        input_path=input_path,
        output_path=output,
        input_grid=grid,
        output_grid=output_grid,
        attrs=attrs if attrs else None,
    )

    typer.echo(f"wrote {result}")


# --------------------------------------------------
# subcommand: info
# --------------------------------------------------
@app.command()
def info(
    path: Path = typer.Argument(..., help="Path to a data file (.h5, .cd, .x, .dat)."),
) -> None:
    """Show metadata summary for a CFD data file."""

    # lazy import to keep CLI startup fast
    from cfd_io.info_mod import get_info

    # extract structured metadata
    file_info = get_info(path)

    # display the result
    typer.echo(f"format:    {file_info.format}")
    typer.echo(f"grid:      {file_info.nx} x {file_info.ny} x {file_info.nz}")
    typer.echo(f"variables: {file_info.np}")
    if file_info.var_names:
        typer.echo(f"  names:   {file_info.var_names}")
    typer.echo(f"timesteps: {file_info.nt}")
    if file_info.timesteps:
        typer.echo(f"  indices: {file_info.timesteps}")
    if file_info.precision:
        typer.echo(f"precision: {file_info.precision}")
