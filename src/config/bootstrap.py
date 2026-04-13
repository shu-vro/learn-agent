from src.config.logger import configure_logging
from loguru import logger
import builtins
import sys

from rich.pretty import install as install_rich_pretty, pretty_repr
from rich.traceback import install as install_rich_traceback

from src.config.env import ENVIRONMENT, LOG_LEVEL

configure_logging(LOG_LEVEL, ENVIRONMENT)

install_rich_pretty()
if ENVIRONMENT.lower() == "development":
    install_rich_traceback(show_locals=True)

# Save original if needed
_original_print = builtins.print


def _format_for_log(value):
    if isinstance(value, str):
        return value
    return pretty_repr(value, expand_all=True)


def print(*args, **kwargs):
    sep = kwargs.pop("sep", " ")
    end = kwargs.pop("end", "\n")
    file = kwargs.pop("file", None)
    flush = kwargs.pop("flush", False)
    log_level = str(kwargs.pop("log_level", "info")).upper()

    # Preserve original print behavior for unsupported kwargs or non-line prints.
    if kwargs or end != "\n":
        _original_print(*args, sep=sep, end=end, file=file, flush=flush)
        return

    target_stream = file
    if target_stream is not None and target_stream not in (sys.stdout, sys.stderr):
        _original_print(*args, sep=sep, end=end, file=target_stream, flush=flush)
        return

    message = sep.join(_format_for_log(arg) for arg in args)

    try:
        logger.opt(depth=1).log(log_level, message)
    except ValueError:
        logger.opt(depth=1).log("INFO", message)


# Override globally
builtins._print = _original_print
builtins.print = print
