"""cfd-io command-line interface (typer).

Subcommands are defined in separate ``cmd_*.py`` modules and
registered on the shared ``app`` Typer instance below.
"""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
from __future__ import annotations

import typer

from .callbacks import verbose_callback, version_callback
from .cmd_attrs import cmd_attrs
from .cmd_convert import cmd_convert
from .cmd_formats import cmd_formats
from .cmd_info import cmd_info

# --------------------------------------------------
# app
# --------------------------------------------------
app = typer.Typer(
    name="cfd-io",
    help="CFD I/O toolkit: read, write, and convert between formats.",
    no_args_is_help=True,
    add_completion=False,
)


# --------------------------------------------------
# callback for --version and --verbose options
# version_callback and verbose_callback are defined in callbacks.py
# --------------------------------------------------
@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=version_callback, is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", callback=verbose_callback,
        help="Enable verbose (debug) output.",
    ),
) -> None:
    """CFD I/O toolkit."""


# --------------------------------------------------
# register subcommands
# notes:
# - the order at which they are registered is the order in which they display
# --------------------------------------------------
app.command(name="formats")(cmd_formats)
app.command(name="info")(cmd_info)
app.command(name="convert")(cmd_convert)
app.command(name="attrs")(cmd_attrs)

# --------------------------------------------------
# main entry point
# --------------------------------------------------
if __name__ == "__main__":
    app()
