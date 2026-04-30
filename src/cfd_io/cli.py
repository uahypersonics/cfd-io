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
    """List supported file formats.

    \b
    Examples:
        # show readers and writers
        cfd-io formats
    """
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
    """Show metadata summary for a CFD data file.

    \b
    Examples:
        # inspect an HDF5 baseflow
        cfd-io info base_flow.h5

        # inspect a Plot3D grid file
        cfd-io info grid.x
    """

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
    if file_info.attrs:
        typer.echo("attributes:")
        for key in sorted(file_info.attrs):
            typer.echo(f"  {key}: {file_info.attrs[key]}")


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
    """

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


# --------------------------------------------------
# subcommand: attrs
# --------------------------------------------------
@app.command()
def attrs(
    path: Path = typer.Argument(..., help="Path to an HDF5 file (.h5, .hdf5)."),
    attr: list[str] = typer.Option(
        ...,
        "--attr", "-a",
        help="Attribute as key=value (repeatable, e.g. -a mach=6.0 -a re1=1e7). "
             "Numeric values are stored as float; otherwise as string.",
    ),
    delete: list[str] = typer.Option(
        [],
        "--delete", "-d",
        help="Attribute key to delete (repeatable).",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging."),
) -> None:
    """Add, update, or delete top-level attributes on an HDF5 file in place.

    Useful for patching flow conditions (mach, re1, temp_inf, ...) onto an
    existing HDF5 dataset that lacks them.

    \b
    Examples:
        # add freestream conditions
        cfd-io attrs base_flow.h5 -a mach=6.0 -a re1=1e7 -a temp_inf=54.0

        # update one attribute and delete another
        cfd-io attrs base_flow.h5 -a mach=10.0 -d stale_key

        # add a string-valued attribute
        cfd-io attrs base_flow.h5 -a case_name=blunt_cone
    """

    # set up logging
    _configure_logging(debug)

    # lazy import of h5py (HDF5-only operation)
    import h5py

    # validate input path
    if not path.exists():
        raise typer.BadParameter(f"file does not exist: {path}")
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise typer.BadParameter(
            f"expected .h5 or .hdf5 file, got '{path.suffix}'"
        )

    # parse --attr key=value pairs into a dict, coercing numerics where possible
    new_attrs: dict[str, float | str] = {}
    for item in attr:
        if "=" not in item:
            raise typer.BadParameter(f"expected key=value, got '{item}'")
        k, v = item.split("=", 1)
        try:
            new_attrs[k] = float(v)
        except ValueError:
            new_attrs[k] = v

    # open in read+write mode and patch attributes
    with h5py.File(path, "r+") as fobj:
        # delete requested keys first
        for key in delete:
            if key in fobj.attrs:
                del fobj.attrs[key]
                typer.echo(f"deleted: {key}")
            else:
                typer.echo(f"skip (not present): {key}")

        # write new/updated attrs
        for key, value in new_attrs.items():
            fobj.attrs[key] = value
            typer.echo(f"set: {key} = {value}")

    typer.echo(f"updated {path}")
