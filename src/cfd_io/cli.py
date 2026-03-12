"""cfd-io CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import logging
from pathlib import Path

import typer

from cfd_io import __version__


# --------------------------------------------------
# cli setup using Typer
# --------------------------------------------------
def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"cfd-io {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="cfd-io",
    help="CFD I/O toolkit: read, write, and convert between formats.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", "-v", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """CFD I/O toolkit."""


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
# subcommand: formats
# --------------------------------------------------
@app.command()
def formats() -> None:
    """List supported file formats."""
    from cfd_io.convert_mod import _READER_REGISTRY, _WRITER_REGISTRY

    readers = " ".join(sorted(_READER_REGISTRY))
    writers = " ".join(sorted(_WRITER_REGISTRY))
    typer.echo(f"Readers:  {readers}")
    typer.echo(f"Writers:  {writers}")


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
        "--grid-in", "-g",
        help="Input grid file (required for split format input).",
    ),
    output_grid: Path | None = typer.Option(
        None,
        "--grid-out", "-G",
        help="Output grid file (required for split format output).",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
    attr: list[str] | None = typer.Option(
        None, "--attr", "-a",
        help="Metadata as key=value (repeatable, e.g. --attr mach=6.0 --attr re=1e6).",
    ),
) -> None:
    """Convert CFD data between formats (dispatched by file extension)."""

    # set up logging
    _configure_logging(debug)

    # lazy import of converter (deferred to keep CLI startup fast)
    from cfd_io.convert_mod import do_convert

    # parse --attr key=value pairs into a dict
    attrs: dict[str, float | str] = {}
    if attr:
        for item in attr:
            if "=" not in item:
                raise typer.BadParameter(f"expected key=value, got '{item}'")
            k, v = item.split("=", 1)
            try:
                attrs[k] = float(v)
            except ValueError:
                attrs[k] = v

    # run the conversion
    result = do_convert(
        input_path=input_path,
        output_path=output,
        input_grid=grid,
        output_grid=output_grid,
        attrs=attrs if attrs else None,
    )

    typer.echo(f"wrote {result}")
