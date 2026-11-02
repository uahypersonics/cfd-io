"""CLI handler for ``cfd-io attrs``."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

import typer

from .helpers import load_attrs_from_file, parse_attr_pairs


# --------------------------------------------------
# subcommand: attrs
# --------------------------------------------------
def cmd_attrs(
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
        new_attrs.update(load_attrs_from_file(attr_file))
    new_attrs.update(parse_attr_pairs(attr))

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
