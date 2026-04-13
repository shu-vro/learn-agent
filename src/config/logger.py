import sys
from loguru import logger


def configure_logging(log_level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> [{time:YYYY-MM-DD HH:mm:ss.SSS Z}] <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
    )
    logger.add(
        "logs/app_{time:YYYY-MM-DD_HH-mm-ss}.log",
        format="{level: <8} [{time:YYYY-MM-DD HH:mm:ss.SSS Z}] {name}:{function}:{line} - {message}",
        level="DEBUG",
        rotation="500 MB",
        retention="1 month",
        backtrace=True,
        diagnose=True,
    )
