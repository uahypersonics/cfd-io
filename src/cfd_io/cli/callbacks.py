"""Callbacks for global options in the cfd-io CLI."""

# --------------------------------------------------
# load necessary modules
# --------------------------------------------------
import logging

import typer


# --------------------------------------------------
# version callback
# --------------------------------------------------
def version_callback(value: bool) -> None:
    """Print version and exit when --version is passed."""
    if value:
        # note: __version__ is defined in cfd_io/__init__.py
        from cfd_io import __version__

        typer.echo(f"cfd-io {__version__}")
        raise typer.Exit()


# --------------------------------------------------
# verbose callback
# --------------------------------------------------
def verbose_callback(value: bool) -> None:
    """Configure normal or verbose (debug) logging to stderr."""

    # select detailed logging only when verbose output is requested
    log_level = logging.DEBUG if value else logging.INFO
    logger = logging.getLogger("cfd_io")
    logger.setLevel(log_level)
    logger.propagate = False

    # reuse the CLI handler when callbacks run repeatedly in one process
    handler = None
    for existing_handler in logger.handlers:
        if getattr(existing_handler, "_cfd_io_cli_handler", False):
            handler = existing_handler
    if handler is None:
        handler = logging.StreamHandler()
        handler._cfd_io_cli_handler = True
        handler.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
        logger.addHandler(handler)

    handler.setLevel(log_level)
