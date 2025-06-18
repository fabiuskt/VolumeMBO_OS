"""
Legacy implementations of volume-preserving clustering logic, kept for benchmarking and fallback purposes.
"""

import warnings

warnings.warn(
    "volumembo.legacy is deprecated and kept for benchmarking or fallback only.",
    DeprecationWarning,
    stacklevel=2,
)

# Expose specific classes/functions if they are reused directly
from .LU_order_statistic import fit_median
