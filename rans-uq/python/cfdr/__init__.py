"""Canonical CFD simulation package; the compiled core is required."""

try:
    from . import _core
except ImportError as error:
    raise ImportError(
        "cfdr requires its compiled _core extension. Build with "
        "-DCFDR_BUILD_PYTHON=ON using the same Python interpreter, or install "
        "the cfdr package; no Python solver fallback is provided."
    ) from error

__all__ = ["_core"]

from .config import PeriodicChannelConfig, SolverConfig

__all__ += ["PeriodicChannelConfig", "SolverConfig"]
