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
    from cfd_io.convert_mod import _READER_REGISTRY, _WRITER_REGISTRY

    readers = " ".join(sorted(_READER_REGISTRY))
    writers = " ".join(sorted(_WRITER_REGISTRY))
    typer.echo(f"Readers:  {readers}")
    typer.echo(f"Writers:  {writers}")
