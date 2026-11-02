"""CLI handler for ``cfd-io info``."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

from pathlib import Path

import typer


# --------------------------------------------------
# subcommand: info
# --------------------------------------------------
def cmd_info(
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
