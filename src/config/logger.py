from loguru import logger


def configure_logging(log_level: str, environment: str = "development") -> None:
    is_development = environment.lower() == "development"

    logger.remove()

    # Console output is handled by Rich through src.config.bootstrap.print().
    logger.add(
        "logs/app_{time:YYYY-MM-DD_HH-mm-ss}.log",
        format="{level: <8} [{time:YYYY-MM-DD HH:mm:ss.SSS Z}] {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="500 MB",
        retention="1 month",
        backtrace=is_development,
        diagnose=is_development,
    )
