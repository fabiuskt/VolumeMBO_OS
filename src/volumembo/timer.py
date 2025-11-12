import time
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np


class TimingManager:
    """
    A class to manage and record timing of code execution.
    This class can be used to measure the time taken by different parts of the code,
    allowing for performance analysis and optimization.
    It supports enabling/disabling timing, resetting recorded timings,
    and summarizing the recorded timings with mean and standard deviation.

    Args:
        enable (bool): If True, timing is enabled. If False, timing is disabled.
                       Defaults to True.
    Usage:
        timing_manager = TimingManager(enable=True)
        with timing_manager.time("some_operation"):
            # Code to time
        summary = timing_manager.summary()
    """

    def __init__(self, enable: bool = True) -> None:
        self.enable: bool = enable
        self.timings: dict[str, list[float]] = {}

    @contextmanager
    def time(self, name: str) -> Iterator[None]:
        """
        Context manager to time a block of code.
        This method can be used to measure the time taken by a specific
        operation or block of code. It records the elapsed time and stores
        it under the given name.
        Args:
            name (str): The name of the operation to be timed.
        """
        if not self.enable:
            yield
            return

        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start

        self.timings.setdefault(name, []).append(elapsed)

    def reset(self, key: str | None = None) -> None:
        """
        Reset the recorded timings.
        This method clears all recorded timings, allowing for fresh measurements.

        Args:
            key (str, optional): If provided, only remove timings for this key.
                                If None, remove all recorded timings.
        """
        if key is None:
            self.timings.clear()
        else:
            self.timings.pop(key, None)

    def summary(self) -> dict[str, dict[str, float | int]]:
        """
        Summarize the recorded timings.
        This method computes the mean, standard deviation, and count of the
        recorded timings for each operation.
        Returns:
            dict: A dictionary where keys are operation names and values are
            dictionaries containing 'mean', 'std', and 'count' of the timings.
        """
        return {
            k: {"mean": np.mean(v), "std": np.std(v), "count": len(v)}
            for k, v in self.timings.items()
        }
