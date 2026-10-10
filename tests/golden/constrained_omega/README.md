# Current SIMPLE regression references

These four generated cases exercise the current default SIMPLE wall constraints.
Each JSON records its source/build identity, mesh, physical inputs, numerical
settings and solve status. The cases are deterministic and use no external data
or random sampling. They are provisional numerical regressions, not physical
validation results.

| Case | Scope |
|---|---|
| `couette_re171_ny96` | Moving-wall Couette flow on the recorded 4x96 inlet/outlet mesh. |
| `periodic_channel_re180` | Body-force planar periodic channel on the recorded 4x56 mesh, using the resolved planar SST wall balance. |
| `bfs_lemoin_uniform_inlet` | Backward-facing step with a uniform inlet and slip top; not matched to a measured-inflow validation case. |
| `cosine_bump_periodic` | Periodic cosine bump; this reference reaches its iteration cap and retains diagnostics only. |

The separate tightly converged planar-channel refinement family does not supply
an error bar for these recorded meshes or stopping tolerances. Curved and
inlet/outlet meshes retain their own wall treatment.

Run the six-case gate with `tests/tools/check.sh --all`, using `CFDR_PYTHON` to select
the interpreter. It records fresh candidates and compares this directory at
rtol 1e-12 and atol 1e-14. Preserve existing generated metadata; regenerate
references only with an explained numerical change and its reviewed comparison.
