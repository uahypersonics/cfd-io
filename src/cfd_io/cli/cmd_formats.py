"""CLI handler for ``cfd-io formats``."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import typer


# --------------------------------------------------
# subcommand: formats
# --------------------------------------------------
def cmd_formats() -> None:
    """List supported file formats.

    \b
    Examples:
        # show readers and writers
        cfd-io formats
    """
    from cfd_io.convert_mod import _READERS, _WRITERS

    readers = " ".join(sorted(_READERS))
    writers = " ".join(sorted(_WRITERS))
    typer.echo(f"Readers:  {readers}")
    typer.echo(f"Writers:  {writers}")
