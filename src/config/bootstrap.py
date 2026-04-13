from src.config.logger import configure_logging
from loguru import logger
import builtins
from pprint import pformat
import sys

from rich.console import Console
from rich.pretty import Pretty, install as install_rich_pretty
from rich.traceback import install as install_rich_traceback

from src.config.env import ENVIRONMENT, LOG_LEVEL

configure_logging(LOG_LEVEL, ENVIRONMENT)

install_rich_pretty()
if ENVIRONMENT.lower() == "development":
    install_rich_traceback(show_locals=True)

_console = Console(stderr=False, soft_wrap=True)
_console_err = Console(stderr=True, soft_wrap=True)

# Save original if needed
_original_print = builtins.print

_LEVEL_STYLES = {
    "TRACE": "dim",
    "DEBUG": "cyan",
    "INFO": "white",
    "SUCCESS": "green",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}


def _format_for_log(value):
    if isinstance(value, str):
        return value
    return pformat(value, compact=True, sort_dicts=False, width=120)


def _format_for_console(value):
    if isinstance(value, str):
        return value
    return Pretty(value)


def print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", None)
    flush = kwargs.pop("flush", False)
    log_level = str(kwargs.pop("log_level", "info")).upper()

    # Fallback for unsupported print kwargs to preserve print compatibility.
    if kwargs:
        _original_print(*args, sep=sep, end=end, file=file, flush=flush)
        return

    message = sep.join(_format_for_log(arg) for arg in args)

    try:
        logger.opt(depth=1).log(log_level, message)
    except ValueError:
        logger.opt(depth=1).log("INFO", message)

    target_stream = file
    if target_stream is not None and target_stream not in (sys.stdout, sys.stderr):
        _original_print(*args, sep=sep, end=end, file=target_stream, flush=flush)
        return

    console = _console_err if target_stream is sys.stderr else _console
    style = _LEVEL_STYLES.get(log_level, "white")
    rich_args = tuple(_format_for_console(arg) for arg in args)
    console.print(*rich_args, sep=sep, end=end, style=style, highlight=True)

    if flush and target_stream is not None:
        target_stream.flush()


# Override globally
builtins._print = _original_print
builtins.print = print
