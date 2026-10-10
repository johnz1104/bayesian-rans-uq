"""Types for the stable Python configuration surface.

The native module also exposes experimental research classes; those APIs retain
pybind11 runtime signatures until their typed wrapper contracts are stabilized.
"""

from typing import Any

class SolverSettings:
    max_iterations: int
    convergence_tol: float
    momentum_convection: str
    turbulence_convection: str
    verbose: bool
    def __init__(self) -> None: ...

class Mesh:
    @staticmethod
    def make_periodic_channel_2d(
        nx: int,
        ny: int,
        Lx: float,
        half_height: float,
        re_tau: float,
        y_plus_target: float = ...,
    ) -> Mesh: ...
    def n_cells(self) -> int: ...
    def compute_wall_distance(self) -> None: ...

def build_info() -> dict[str, str]: ...
def __getattr__(name: str) -> Any: ...
