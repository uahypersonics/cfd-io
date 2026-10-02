"""cfd-io CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import json
import logging
import re
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
# helper: load attributes from a flow_conditions file
# --------------------------------------------------
def _load_attrs_from_file(path: Path) -> dict[str, float | str]:
    """Load attributes from a flow conditions file (.dat legacy or .json).

    .json format: flat {"key": value} dict — values cast to float where possible.

    .dat legacy format: fixed-column lines like
        ``freestream Mach number, M      :    14.4  [-]``
    The short name after the last comma is used as the key (lowercased).
    For lines without a comma short-name the full description is snake_cased.
    The first numeric token before any ``[units]`` block is used as the value.
    String-valued lines (e.g. viscosity law name) are stored as strings.

    Args:
        path: Path to the attribute file (.dat or .json).

    Returns:
        Dict mapping attribute keys to float or string values.
    """
    # validate file exists
    if not path.exists():
        raise typer.BadParameter(f"attr-file not found: {path}")

    # read JSON format
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # cast values to float where possible
        attrs: dict[str, float | str] = {}
        for k, v in data.items():
            try:
                attrs[k] = float(v)
            except (ValueError, TypeError):
                attrs[k] = str(v)
        return attrs

    # read legacy .dat format
    if suffix == ".dat":
        attrs = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # skip blank lines and lines without a colon separator
                if not line or ":" not in line:
                    continue

                # split on first colon: left = description, right = value + units
                desc, _, value_part = line.partition(":")
                desc = desc.strip()
                value_part = value_part.strip()

                # extract key: last token after last comma in description (the short name)
                # example: "freestream Mach number, M" -> "m"
                # example: "Viscosity Option" -> "viscosity_option" (no comma)
                if "," in desc:
                    key = desc.rsplit(",", 1)[-1].strip().lower()
                else:
                    # snake_case the full description as fallback
                    key = re.sub(r"[^a-z0-9]+", "_", desc.lower()).strip("_")

                # skip if key is empty
                if not key:
                    continue

                # extract numeric value: strip unit block [...]  and dual-value slash
                # example: "14.4000 [-]"         -> "14.4000"
                # example: "119161873.55 / 17282.97 [Pa]/[psi]" -> "119161873.55"
                raw = value_part.split("[")[0].split("/")[0].strip()

                # store as float if numeric, otherwise as a stripped string
                try:
                    attrs[key] = float(raw)
                except ValueError:
                    attrs[key] = value_part.split("[")[0].strip()

        return attrs

    raise typer.BadParameter(
        f"unsupported attr-file format '{suffix}' — expected .dat or .json"
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
    path: Path = typer.Argument(..., help="Path to a supported CFD data file."),
) -> None:
    """Show metadata summary for a CFD data file.

    \b
    Examples:
        # inspect an HDF5 baseflow
        cfd-io info base_flow.h5

        # inspect a Plot3D grid file
        cfd-io info grid.x

        # inspect an SU2 VTU base flow
        cfd-io info flow_data.vtu
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

    # set up logging
    _configure_logging(debug)

    # lazy import of converter (deferred to keep CLI startup fast)
    from cfd_io.convert_mod import do_convert

    # load attributes from file first (lower precedence)
    attrs: dict[str, float | str] = {}
    if attr_file:
        attrs.update(_load_attrs_from_file(attr_file))

    # parse --attr key=value pairs (these override any file-loaded attrs)
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
        [],
        "--attr", "-a",
        help="Attribute as key=value (repeatable, e.g. -a mach=6.0 -a re1=1e7). "
             "Numeric values are stored as float; otherwise as string.",
    ),
    delete: list[str] = typer.Option(
        [],
        "--delete", "-d",
        help="Attribute key to delete (repeatable).",
    ),
    attr_file: Path | None = typer.Option(
        None, "--attr-file", "-A",
        help="Load attributes from a file (.dat legacy or .json). "
             "Merged with --attr; --attr takes precedence on conflicts.",
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

        # stamp all conditions from a legacy flow_conditions.dat
        cfd-io attrs base_flow.h5 -A flow_conditions.dat

        # stamp from JSON, override one value
        cfd-io attrs base_flow.h5 -A flow_conditions.json -a mach=6.0

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
    # load attributes from file first (lower precedence), then overlay --attr
    new_attrs: dict[str, float | str] = {}
    if attr_file:
        new_attrs.update(_load_attrs_from_file(attr_file))
    for item in attr:
        if "=" not in item:
            raise typer.BadParameter(f"expected key=value, got '{item}'")
        k, v = item.split("=", 1)
        try:
            new_attrs[k] = float(v)
        except ValueError:
            new_attrs[k] = v

    # require at least one attribute source
    if not new_attrs and not delete:
        raise typer.BadParameter("provide at least one --attr, --attr-file, or --delete")

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
