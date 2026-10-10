"""Small immutable configurations for the native solver entry points."""

import math
from dataclasses import dataclass
from typing import Literal

from . import _core


@dataclass(frozen=True)
class SolverConfig:
    """Common SIMPLE controls; other native controls remain explicitly accessible."""

    max_iterations: int = 500
    convergence_tol: float = 1e-5
    convection: Literal["upwind", "linearUpwind", "limitedLinearUpwind"] = "limitedLinearUpwind"
    verbose: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.max_iterations, bool) or not isinstance(self.max_iterations, int):
            raise TypeError("max_iterations must be an integer")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if not math.isfinite(self.convergence_tol) or self.convergence_tol <= 0:
            raise ValueError("convergence_tol must be finite and positive")
        if self.convection not in {"upwind", "linearUpwind", "limitedLinearUpwind"}:
            raise ValueError("unknown convection scheme")

    def native(self) -> _core.SolverSettings:
        settings = _core.SolverSettings()
        settings.max_iterations = self.max_iterations
        settings.convergence_tol = self.convergence_tol
        settings.momentum_convection = self.convection
        settings.turbulence_convection = self.convection
        settings.verbose = self.verbose
        return settings


@dataclass(frozen=True)
class PeriodicChannelConfig:
    """Geometry only: callers supply forcing, viscosity and scientific run controls."""

    nx: int = 4
    ny: int = 16
    length: float = 2.0
    half_height: float = 1.0
    re_tau: float = 180.0
    y_plus_target: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("nx", self.nx), ("ny", self.ny)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < (3 if name == "nx" else 2):
                raise ValueError(f"{name} is too small for a periodic channel")
        for size in (self.length, self.half_height, self.y_plus_target):
            if not math.isfinite(size) or size <= 0:
                raise ValueError(
                    "length, half_height and y_plus_target must be finite and positive"
                )
        if not math.isfinite(self.re_tau) or self.re_tau <= 0:
            raise ValueError("re_tau must be finite and positive")

    def mesh(self) -> _core.Mesh:
        mesh = _core.Mesh.make_periodic_channel_2d(
            self.nx, self.ny, self.length, self.half_height, self.re_tau, self.y_plus_target
        )
        mesh.compute_wall_distance()
        return mesh
