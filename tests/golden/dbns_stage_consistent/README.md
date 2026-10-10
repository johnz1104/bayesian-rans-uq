# Current DBNS regression references

These generated deterministic cases exercise density-based compressible flow.
Each JSON preserves its recorded source/build identity, mesh and numerical
settings. No external data or random sampling is used. These are provisional
numerical regressions, not general compressible-flow validation.

| Case | Scope |
|---|---|
| `dbns_shock_tubes` | Sod and Lax shock tubes on 200x1 strips, compared with exact Riemann solutions at their requested final times. |
| `dbns_mms` | Steady inviscid Euler manufactured solution on 16x16, 32x32 and 64x64 grids, with recorded density errors and refinement orders. |

The shock records include completed-step accounting. The MMS generator does
not record each individual steady status, so that JSON alone does not certify
converged roots. Separate outcome tests check stopping and iterative error.
Neither case certifies turbulent, nonorthogonal or fully coupled implicit flow.

`tests/tools/check.sh --all` records fresh candidates and compares this directory
at rtol 1e-12 and atol 1e-14. It uses a separate candidate directory for the
four SIMPLE references. Preserve generated metadata and retain the physical
scope when interpreting a passing comparison.
