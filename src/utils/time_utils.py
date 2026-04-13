from contextlib import contextmanager
from functools import wraps
import time
from typing import Any, Callable, Iterator


@contextmanager
def _measure_time_context(
    label: str,
    tracker: dict[str, float] | None = None,
    log_level: str = "success",
) -> Iterator[None]:
    """Context manager that measures elapsed time and logs in a consistent format."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        elapsed_time = time.perf_counter() - start_time
        if tracker is not None:
            tracker[label] = elapsed_time
        print(
            f"Execution time of {label}: {elapsed_time:.2f} seconds",
            log_level=log_level,
        )


def measure_time(
    name_or_func: str | Callable[..., Any],
    *,
    tracker: dict[str, float] | None = None,
    log_level: str = "success",
):
    """Measure execution time as either a decorator or a context manager.

    Usage:
    - Decorator: @measure_time
    - Context manager: with measure_time("step_name", tracker=tracker):

    Example:
    ```python
    @measure_time
    def my_function():
        # function body

    # or
    tracker = {}
    with measure_time("my_step", tracker=tracker):
        # code block
    ```
    """

    if callable(name_or_func):
        func = name_or_func

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any):
            with _measure_time_context(
                func.__name__,
                tracker=tracker,
                log_level=log_level,
            ):
                return func(*args, **kwargs)

        return wrapper

    return _measure_time_context(
        str(name_or_func),
        tracker=tracker,
        log_level=log_level,
    )
