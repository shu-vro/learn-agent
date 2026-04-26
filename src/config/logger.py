from loguru import logger
from rich.logging import RichHandler


def configure_logging(log_level: str, environment: str = "development") -> None:
    is_development = environment.lower() == "development"

    logger.remove()

    rich_handler = RichHandler(
        rich_tracebacks=is_development,
        tracebacks_show_locals=is_development,
        show_time=False,
        show_level=False,
        show_path=False,
        markup=False,
    )

    logger.add(
        rich_handler,
        level=log_level,
        format="[{level}] {name}:{function}:{line} - {message}",
        backtrace=is_development,
        diagnose=is_development,
        colorize=False,
    )

    logger.add(
        "logs/app_{time:YYYY-MM-DD_HH-mm-ss}.log",
        format="<{time:YYYY-MM-DD HH:mm:ss.SSS Z}> [{level}] {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="500 MB",
        retention="1 month",
        backtrace=is_development,
        diagnose=is_development,
    )
