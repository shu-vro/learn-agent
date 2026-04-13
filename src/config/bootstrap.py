from src.config.logger import configure_logging
from loguru import logger
import builtins
from src.config.env import LOG_LEVEL

configure_logging(LOG_LEVEL)

# Save original if needed
_original_print = builtins.print


def print(*args, **kwargs):
    sep = kwargs.get("sep", " ")
    log_level = str(kwargs.get("log_level", "info")).upper()
    message = sep.join(str(a) for a in args)
    try:
        logger.opt(depth=1).log(log_level, message)
    except ValueError:
        logger.opt(depth=1).log("INFO", message)


# Override globally
builtins._print = _original_print
builtins.print = print
