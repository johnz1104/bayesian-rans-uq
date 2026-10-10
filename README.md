# bayesian-rans-uq

Bayesian calibration of Reynolds-averaged Navier–Stokes (RANS) turbulence models tends to produce overconfident predictions: the posterior concentrates on a single best-fitting set of closure coefficients, and the reported uncertainty is too narrow for flow types outside the calibration set. This library provides an end-to-end framework for performing such calibrations and assessing their reliability. It couples a finite-volume RANS solver with observation operators and a statistical layer for sampling, importance weighting and predictive scoring. Solver convergence is checked for every evaluation, and failed solves are excluded from the posterior, so that numerical error is not mistaken for model uncertainty.

## Capabilities

- A steady two-dimensional incompressible flow solver based on the SIMPLE algorithm, with Rhie–Chow pressure–velocity coupling and the SST k–ω turbulence model.
- Observation operators that map a solution to measurable quantities: velocity and pressure at arbitrary points in the domain, reconstructed linearly from cell-centered values, and wall pressure and skin-friction coefficients, each paired with a Gaussian likelihood.
- An explicit convergence status for every evaluation. Failed and iteration-capped solves are retained and labelled, and are excluded from the posterior.
- Parallel batch evaluation of coefficient samples on shared-memory worker threads, with bitwise-identical results across worker counts and sample orderings.
- A statistical layer for importance-sampling calibration: reproducible prior designs with stable sample identifiers and seeded random streams, likelihood-based importance weights, posterior resampling, and predictive assessment by the continuous ranked probability score (CRPS), split-conformal prediction intervals and rank histograms.
- Experimental components: steady-state sensitivities of observations with respect to the SST coefficients, and a density-based compressible solver (HLLC fluxes with MUSCL reconstruction) in `extras/`.

## Repository structure

```
core/                mesh, fields, linear solvers, SST model, observations
rans-uq/             SIMPLE solver, forward models, sensitivities, batches
  python/            pybind11 bindings and the Python package
extras/
  compressible/      density-based solver and an experimental low-Mach solver
tests/               test registration, reference solutions, check scripts
cmake/               build and install configuration
docs/                installation details
```

The Python package and the CMake targets are named `cfdr`.

## Installation

The library requires Python 3.11 or later, a C++17 compiler and CMake 3.21 or later. The Python package is installed with:

```sh
python3 -m venv .venv
. .venv/bin/activate
CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install .
```

The C++ libraries alone are installed with:

```sh
cmake -S . -B build/native -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF -DCFDR_BUILD_PYTHON=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/build/prefix"
cmake --build build/native --parallel 2
cmake --install build/native
```

Further details are given in [docs/installation.md](docs/installation.md).

## Example

The following example performs a complete calibration on a deliberately small problem: a 3×4-cell channel, a single synthetic velocity measurement and the SST coefficients a1 and β*. It runs in a few seconds.

```python
import numpy as np
from cfdr import _core as c
from cfdr import statistics as stats

# A small channel with one synthetic measurement: u = 0.8 ± 0.1 at (0.5, 0.4).
mesh = c.Mesh.make_channel_2d(3, 4, 1.0, 1.0)
mesh.compute_wall_distance()
observations = c.ObservationOperator()
observations.add_velocity_profile(c.Vec3(0.5, 0.4, 0.0), 0, 0.8, 0.1)
boundaries = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 0.02, 3.0)
settings = c.SolverSettings()
settings.verbose = False
settings.max_iterations = 5000
settings.convergence_tol = 1e-5
batch = c.ForwardBatch(
    mesh, c.InferenceParameterSet.a1_betaStar(), observations, boundaries, 0.01, settings
)

# Uniform prior draws for (a1, beta*), solved on two worker threads.
prior = stats.independent_uniform_design(range(24), [0.30, 0.085], [0.34, 0.095], root_seed=1)
evaluations = stats.evaluate_batch(prior, batch, workers=2)

# Importance weights, then a posterior predictive for the measured velocity.
weights = stats.posterior_weights(evaluations)
theta = np.array([record.theta for record in evaluations.records])
print("posterior mean of (a1, beta*):", np.average(theta, axis=0, weights=weights))

posterior = stats.resample_posterior(evaluations, range(1000, 1024), root_seed=1)
predicted = stats.predictive_summary(
    stats.evaluate_batch(posterior, batch, workers=2), root_seed=1, observation_sigma=(0.1,)
)
print(f"predicted u: {predicted.latent_mean[0]:.4f} ± {np.sqrt(predicted.total_variance[0]):.4f}")
print(f"  of which from the coefficients: ± {np.sqrt(predicted.latent_variance[0]):.1e}")
```

Expected output:

```
Channel2D mesh: 3x4 (12 cells, 31 faces)
posterior mean of (a1, beta*): [0.31792365 0.0896799 ]
predicted u: 1.2116 ± 0.1000
  of which from the coefficients: ± 6.9e-07
```

The predicted velocity (1.21) lies four standard deviations from the measurement (0.8). On a mesh this coarse, a1 and β* have a negligible effect on the observed velocity, so the importance weights remain uniform and the posterior coincides with the prior; nearly all of the reported ±0.10 is the assumed measurement noise. Nothing in the posterior indicates that no coefficient values can reproduce the data. This is the overconfidence problem in miniature, and it motivates reporting the predictive variance by source, as the predictive summary does.

If any solve fails to converge, `posterior_weights` raises an error that reports the number of failures rather than silently discarding them.

C++ applications link the installed CMake targets:

```cmake
cmake_minimum_required(VERSION 3.21)
project(cfdr_cmake_consumer LANGUAGES CXX)
find_package(cfdr 0.1 CONFIG REQUIRED)
add_executable(cmake_consumer main.cpp)
target_link_libraries(cmake_consumer PRIVATE cfdr::incompressible cfdr::dbns)
enable_testing()
add_test(NAME cmake_consumer COMMAND cmake_consumer)
```

A complete example is provided in [tests/install-consumer](tests/install-consumer/main.cpp). After the C++ installation above, it is built and run with:

```sh
cmake -S tests/install-consumer -B build/example \
  -DCMAKE_PREFIX_PATH="$PWD/build/prefix"
cmake --build build/example --parallel 2
ctest --test-dir build/example --output-on-failure
```

## Current work

- The SIMPLE solver has no non-orthogonal correction, so its accuracy on curved meshes, such as periodic-hill meshes finer than 60×40 cells, has not been established.
- The backward-facing step and periodic-hill cases have no matched, grid-converged reference comparison, so they have not yet been validated.
- The calibration layer has no Markov chain Monte Carlo sampler, so posteriors are formed by importance weighting of independent prior draws.
- The steady-state sensitivity method has been verified on a single small case, so it remains experimental.
- The density-based solver in `extras/` uses a simplified SST diffusion model, so its turbulent and implicit paths remain experimental. The low-Mach solver omits general compressible energy terms.
- The solvers run on a single node, with thread-level parallelism for batch evaluations. Distributed-memory parallelization with MPI is in development, beginning with ensemble batches distributed across ranks and followed by domain-decomposed SIMPLE.
