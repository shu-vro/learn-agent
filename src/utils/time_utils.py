from functools import wraps
import time


def measure_time(func):
    """Decorator to measure the execution time of a function."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Execution time of {func.__name__}: {elapsed_time:.2f} seconds",
            log_level="success",
        )
        return result

    return wrapper
