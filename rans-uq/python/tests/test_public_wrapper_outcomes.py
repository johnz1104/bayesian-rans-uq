"""Bounded values and ownership contracts for maintained non-DBNS wrappers.

Deterministic synthetic fixtures, without external data or random sampling.
These are interface outcomes, not broader solver or sensitivity certification.
"""

import gc

import numpy as np
import pytest

from cfdr import _core as c


def channel():
    # With two wall-normal cells the symmetric generator gives unit cubes.
    mesh = c.Mesh.make_channel_2d(3, 2, 3.0, 2.0)
    mesh.compute_wall_distance()
    return mesh


def settings(iterations=1):
    value = c.SolverSettings()
    value.verbose = False
    value.max_iterations = iterations
    value.turb_start_iter = 10
    value.nut_floor_iters = 0
    value.momentum_convection = value.turbulence_convection = "upwind"
    return value


@pytest.mark.parametrize(
    ("stress", "expected"),
    [
        ([2, 2, 2, 0, 0, 0], [2, 2, 2, 0, 0, 0]),
        ([1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]),
        ([2, -0.5, -0.5, 0, 0, 0], [1, 0, 0, 0, 0, 0]),
        # Rotate the preceding indefinite tensor onto v=(.6,.8,0).
        ([0.4, 1.1, -0.5, 1.2, 0, 0], [0.36, 0.64, 0, 0.48, 0, 0]),
    ],
)
def test_stress_projection_has_known_components_and_trace(stress, expected):
    result = c.project_reynolds_stress(*stress)
    np.testing.assert_allclose(result, expected, rtol=0, atol=2e-14)
    assert sum(result[:3]) == pytest.approx(sum(stress[:3]), abs=2e-14)


def test_mesh_exports_match_unit_cube_geometry_and_own_storage():
    mesh = channel()
    expected_cells = [[x, y, 0.5] for y in (0.5, 1.5) for x in (0.5, 1.5, 2.5)]
    expected_nodes = [[x, y, z] for z in (0, 1) for y in (0, 1, 2) for x in (0, 1, 2, 3)]
    # Compare geometric sets; face ordering is not needed by this public check.
    expected_faces = [[x, y, 0.5] for y in (0, 1, 2) for x in (0.5, 1.5, 2.5)] + [
        [x, y, 0.5] for y in (0.5, 1.5) for x in (0, 1, 2, 3)
    ]
    arrays = [mesh.cell_centers(), mesh.cell_volumes(), mesh.node_coords(), mesh.face_centers()]
    np.testing.assert_array_equal(arrays[0], expected_cells)
    np.testing.assert_array_equal(arrays[1], np.ones(6))
    np.testing.assert_array_equal(arrays[2], expected_nodes)
    assert sorted(map(tuple, arrays[3])) == sorted(map(tuple, expected_faces))
    for value in arrays:
        assert value.dtype == np.float64
        assert value.flags.owndata and value.flags.c_contiguous
    arrays[0][:] = -19
    np.testing.assert_array_equal(mesh.cell_centers(), expected_cells)
    names, types = mesh.patch_names(), mesh.patch_types()
    assert dict(zip(names, types)) == {
        "bottom": "wall",
        "top": "wall",
        "inlet": "inlet",
        "outlet": "outlet",
    }
    types[0] = "changed copy"
    assert mesh.patch_types()[0] == "wall"
    del mesh
    gc.collect()
    np.testing.assert_array_equal(arrays[2], expected_nodes)


@pytest.mark.parametrize(
    ("name", "owners", "normal", "centers"),
    [
        ("bottom", [0, 1, 2], [0, -1, 0], [[x, 0, 0.5] for x in (0.5, 1.5, 2.5)]),
        ("top", [3, 4, 5], [0, 1, 0], [[x, 2, 0.5] for x in (0.5, 1.5, 2.5)]),
        ("inlet", [0, 3], [-1, 0, 0], [[0, y, 0.5] for y in (0.5, 1.5)]),
        ("outlet", [2, 5], [1, 0, 0], [[3, y, 0.5] for y in (0.5, 1.5)]),
    ],
)
def test_patch_data_has_expected_geometry_and_independent_arrays(name, owners, normal, centers):
    mesh = channel()
    patch = mesh.wall_patch_data(name)
    assert patch["name"] == name and patch["n_faces"] == len(owners)
    assert patch["type"] == ("wall" if name in {"bottom", "top"} else name)
    np.testing.assert_array_equal(patch["owner"], owners)
    np.testing.assert_array_equal(patch["normal"], np.tile(normal, (len(owners), 1)))
    np.testing.assert_array_equal(patch["center"], centers)
    np.testing.assert_array_equal(patch["delta"], np.full(len(owners), 0.5))
    np.testing.assert_array_equal(patch["area"], np.ones(len(owners)))
    assert patch["owner"].dtype.kind == "i"
    assert all(patch[key].flags.owndata for key in ("owner", "normal", "center", "delta", "area"))
    patch["owner"][:] = -1
    np.testing.assert_array_equal(mesh.wall_patch_data(name)["owner"], owners)
    with pytest.raises(RuntimeError, match="unknown patch"):
        mesh.wall_patch_data("missing")
    del mesh
    gc.collect()
    np.testing.assert_array_equal(patch["center"], centers)


def test_curved_generator_converts_strided_arrays_and_copies_geometry():
    x = np.array([0, -99, 1, -99, 3, -99, 4, -99], dtype=np.float32)[::2]
    bottom = np.full(8, 0.25, dtype=np.float32)[::-2]
    mesh = c.Mesh.make_curved_channel_periodic_2d(x, bottom, 2.25, 2, 0.0)
    x[:] = bottom[:] = np.nan
    expected = [[x, y, 0.5] for y in (0.75, 1.75) for x in (0.5, 2.0, 3.5)]
    np.testing.assert_allclose(mesh.cell_centers(), expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(mesh.cell_volumes(), [1, 2, 1, 1, 2, 1], rtol=0, atol=1e-14)
    assert mesh.patch_types() == ["wall", "wall"]
    for bad_x, bad_bottom in [(np.zeros((4, 1)), np.zeros(4)), (np.arange(4), np.zeros(3))]:
        with pytest.raises((ValueError, RuntimeError)):
            c.Mesh.make_curved_channel_periodic_2d(bad_x, bad_bottom, 2.25, 2, 0.0)


def inlet_snapshot(mesh, bcs):
    solver = c.SIMPLESolver(mesh, c.SSTCoefficients(), bcs, 0.1, settings())
    fields = c.FlowFields(mesh)
    solver.init_uniform(fields, c.Vec3(0, 0, 0), 0, 0.02, 3.0)
    boundary = fields.snapshot()["boundary"]
    inlet = mesh.face_centers()[mesh.n_internal_faces() :, 0] == 0
    return {name: boundary[name][inlet] for name in ("U", "k", "omega")}


def test_boundary_profiles_copy_strided_values_and_reject_without_replacing():
    mesh = channel()
    bcs = c.FlowBoundaryConditions.channel_defaults(mesh, 0.0, 0.02, 3.0)
    velocity = np.array([[1.0, 0.1, 0.2], [2.0, 0.3, 0.4]], dtype=np.float32)
    velocity_input = np.asfortranarray(velocity)[::-1]
    k_input = np.array([0.06, -99, 0.03, -99])[::2]
    omega_input = np.array([7.0, 5.0])[::-1]
    expected = {"U": velocity_input.copy(), "k": k_input.copy(), "omega": omega_input.copy()}
    bcs.set_velocity_profile(mesh, "inlet", velocity_input)
    bcs.set_k_profile(mesh, "inlet", k_input)
    bcs.set_omega_profile(mesh, "inlet", omega_input)
    velocity_input[:] = k_input[:] = omega_input[:] = np.nan
    for name, values in inlet_snapshot(mesh, bcs).items():
        np.testing.assert_array_equal(values, expected[name])
    methods = [
        (bcs.set_velocity_profile, "U"),
        (bcs.set_k_profile, "k"),
        (bcs.set_omega_profile, "omega"),
    ]
    for setter, name in methods:
        valid = expected[name]
        malformed = [valid[:1], valid.reshape((1,) + valid.shape), np.full_like(valid, np.nan)]
        if name == "U":
            malformed.append(np.ones((2, 2)))
        for bad in malformed:
            with pytest.raises((ValueError, RuntimeError)):
                setter(mesh, "inlet", bad)
        with pytest.raises((ValueError, RuntimeError)):
            setter(mesh, "missing", valid)
    for name, values in inlet_snapshot(mesh, bcs).items():
        np.testing.assert_array_equal(values, expected[name])


def test_simple_injection_scale_and_clear_have_expected_source_outcomes():
    mesh = channel()
    bcs = c.FlowBoundaryConditions.channel_defaults(mesh, 0.0, 0.02, 3.0)
    solver = c.SIMPLESolver(mesh, c.SSTCoefficients(), bcs, 0.1, settings())
    target = np.zeros((6, 6))
    target[:, 0] = 0.1 * mesh.cell_centers()[:, 0]
    target[:, 1] = -target[:, 0]
    solver.set_target_anisotropy(target)
    expected = np.zeros((6, 3))
    # -div(2*k*b), k=.02: interior central difference and owner-extrapolated ends.
    expected[:, 0] = [-0.002, -0.004, -0.002] * 2
    for scale in (1.0, 0.5, 0.0):
        solver.set_injection_scale(scale)
        fields = c.FlowFields(mesh)
        solver.init_uniform(fields, c.Vec3(0, 0, 0), 0, 0.02, 3)
        snapshots = []
        solver.solve(fields, source_observer=snapshots.append)
        assert len(snapshots) == 1
        np.testing.assert_allclose(
            snapshots[0]["fresh_integrated_force"], scale * expected, atol=1e-15
        )
        np.testing.assert_allclose(
            snapshots[0]["applied_integrated_force"], scale * expected, atol=1e-15
        )
    for invalid in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="scale"):
            solver.set_injection_scale(invalid)
    solver.set_injection_scale(1.0)
    solver.clear_target_anisotropy()
    snapshots = []
    solver.init_uniform(fields, c.Vec3(0, 0, 0), 0, 0.02, 3)
    solver.solve(fields, source_observer=snapshots.append)
    np.testing.assert_array_equal(snapshots[0]["fresh_integrated_force"], np.zeros((6, 3)))
    np.testing.assert_array_equal(snapshots[0]["applied_integrated_force"], np.zeros((6, 3)))


def test_forward_target_symmetrizes_copies_and_returns_injection_diagnostics():
    mesh = channel()
    bcs = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 0.02, 3.0)
    params = c.InferenceParameterSet.a1_betaStar()
    obs = c.ObservationOperator()
    control = settings(2)
    fm = c.ForwardModel(mesh, params, obs, bcs, 0.1, control, c.Vec3(1, 0, 0), 0, 0.02, 3)
    target = np.zeros((6, 3, 6))[:, :, ::2]
    x = mesh.cell_centers()[:, 0]
    target[:, 0, 0], target[:, 1, 1] = 0.02 * x, -0.02 * x
    target[:, 0, 1], target[:, 1, 0] = 0.09 * x, -0.03 * x
    expected_six = np.zeros((6, 6))
    expected_six[:, 0], expected_six[:, 1], expected_six[:, 3] = 0.02 * x, -0.02 * x, 0.03 * x
    fm.set_target_anisotropy(target)
    target[:] = np.nan
    assert fm.has_target_anisotropy()
    for malformed in (
        np.zeros((6, 3)),
        np.zeros((6, 3, 2)),
        np.zeros((5, 3, 3)),
        np.full((6, 3, 3), np.nan),
    ):
        with pytest.raises((RuntimeError, ValueError)):
            fm.set_target_anisotropy(malformed)
    result = fm.evaluate(params.pack(c.SSTCoefficients()))
    assert result.status == c.EvaluationStatus.Unconverged and result.simple_iters == 2
    reference = c.SIMPLESolver(mesh, c.SSTCoefficients(), bcs, 0.1, control)
    fields = c.FlowFields(mesh)
    reference.init_uniform(fields, c.Vec3(1, 0, 0), 0, 0.02, 3)
    reference.set_target_anisotropy(expected_six)
    reference.solve(fields)
    snapshot = fields.snapshot()
    for name, values in fm.last_fields().items():
        np.testing.assert_allclose(values, snapshot[name], rtol=0, atol=1e-14)
    # A constant shear target could conceal a conversion error behind zero
    # divergence. This varying target must distinguish taking just b_xy from
    # the specified average of b_xy and b_yx.
    unsymmetrized = expected_six.copy()
    unsymmetrized[:, 3] = 0.09 * x
    negative = c.SIMPLESolver(mesh, c.SSTCoefficients(), bcs, 0.1, control)
    negative_fields = c.FlowFields(mesh)
    negative.init_uniform(negative_fields, c.Vec3(1, 0, 0), 0, 0.02, 3)
    negative.set_target_anisotropy(unsymmetrized)
    negative.solve(negative_fields)
    assert not np.allclose(snapshot["U"], negative_fields.snapshot()["U"], rtol=1e-10, atol=1e-12)
    diagnostics = fm.injection_diagnostics()
    assert diagnostics == {
        "active": True,
        "checked_iters": 2,
        "all_realizable": True,
        "max_violation": 0.0,
    }
    diagnostics["active"] = False
    assert fm.injection_diagnostics()["active"]
    fm.clear_target_anisotropy()
    assert not fm.has_target_anisotropy()
    fm.cache().set_byte_budget(0)
    fm.evaluate(params.pack(c.SSTCoefficients()))
    assert not fm.injection_diagnostics()["active"]


def test_odd_even_ratio_has_exact_modes_and_rejects_wrong_length():
    mesh = channel()
    assert c.odd_even_energy_ratio(mesh, [3.0] * 6) == 0
    assert c.odd_even_energy_ratio(mesh, [1, -1, 1, -1, 1, -1]) == pytest.approx(1.0)
    assert c.odd_even_energy_ratio(mesh, [1, 0, -1, 1, 0, -1]) == pytest.approx(0.0, abs=1e-15)
    for values in ([1.0] * 5, [1.0] * 7):
        with pytest.raises(ValueError, match="exactly one value"):
            c.odd_even_energy_ratio(mesh, values)


def test_steady_base_owns_fields_metadata_and_survives_invalidated_state():
    # Asymmetric 3x4 developing outlet with resolved SST and upwind convection.
    # This checks the primal snapshot, without requesting a tangent solve.
    mesh = c.Mesh.make_channel_2d(3, 4, 1.0, 1.0)
    mesh.compute_wall_distance()
    bcs = c.FlowBoundaryConditions.channel_defaults(mesh, 1.0, 0.01, 3.0)
    y = mesh.wall_patch_data("inlet")["center"][:, 1]
    inlet = np.column_stack((0.7 + 0.4 * y, 0.06 * (0.2 + y), np.zeros_like(y)))
    bcs.set_velocity_profile(mesh, "inlet", inlet)
    control = c.SolverSettings()
    control.verbose = False
    control.max_iterations = 20000
    control.convergence_tol = control.inner_tolerance = 1e-10
    control.inner_iterations = 400
    control.nut_floor_iters = 0
    control.momentum_convection = control.turbulence_convection = "upwind"
    obs = c.ObservationOperator()
    obs.add_velocity_profile(c.Vec3(0.63, 0.41, 0), 0, 0, 1)
    sensitivity = c.ParameterSensitivity(mesh, obs, bcs, 0.003, control)
    theta = c.InferenceParameterSet.all11().pack(c.SSTCoefficients())
    # n_state counts only Ux/Uy/k/omega. The augmented steady
    # vector below additionally contains pressure and free-face fluxes.
    assert (sensitivity.n_cells(), sensitivity.n_state(), sensitivity.n_obs()) == (12, 48, 1)
    with pytest.raises(RuntimeError, match="converged base"):
        sensitivity.steady_base()
    assert sensitivity.solve_state(theta) == c.EvaluationStatus.Converged
    base = sensitivity.steady_base()
    assert base["coefficients"] == theta
    assert base["molecular_viscosity"] == 0.003
    assert base["source_digest"] == c.build_info()["source_digest"]
    assert base["history"].converged and not base["history"].diverged
    assert base["settings"].convergence_tol == 1e-10
    assert len(base["corrected_fluxes"]) == mesh.n_faces()
    q = sensitivity.steady_state_vector()
    fields = base["fields"]
    for index, values in enumerate(
        (fields["U"][:, 0], fields["U"][:, 1], fields["k"], fields["omega"], fields["p"])
    ):
        np.testing.assert_array_equal(values, q[12 * index : 12 * (index + 1)])
    # Independently use the copied boundary object: it must carry the fixture's
    # prescribed asymmetric inlet, not a default or stale boundary definition.
    solver = c.SIMPLESolver(mesh, c.SSTCoefficients(), base["boundaries"], 0.003, control)
    restored = c.FlowFields(mesh)
    solver.init_uniform(restored, c.Vec3(1, 0, 0), 0, 1e-4, 1)
    inlet_faces = mesh.face_centers()[mesh.n_internal_faces() :, 0] == 0
    np.testing.assert_array_equal(restored.snapshot()["boundary"]["U"][inlet_faces], inlet)
    assert solver.solve(restored).converged
    np.testing.assert_allclose(base["corrected_fluxes"], solver.face_fluxes(), rtol=0, atol=1e-14)
    for name in ("U", "p", "k", "omega", "nuT", "F1", "F2", "Pk", "CDkw"):
        np.testing.assert_allclose(fields[name], restored.snapshot()[name], rtol=0, atol=1e-14)
    assert all(fields[name].flags.owndata for name in ("U", "p", "k", "omega"))
    assert all(value.flags.owndata for value in fields["boundary"].values())
    saved = sensitivity.steady_base()
    fields["U"][:] = -99
    base["coefficients"][0] = -99
    base["corrected_fluxes"][0] = -99
    base["settings"].convergence_tol = 0.5
    fresh = sensitivity.steady_base()
    np.testing.assert_array_equal(fresh["fields"]["U"], saved["fields"]["U"])
    assert fresh["coefficients"] == theta
    assert fresh["corrected_fluxes"] == saved["corrected_fluxes"]
    assert fresh["settings"].convergence_tol == 1e-10
    assert sensitivity.solve_state([np.nan] * 11) == c.EvaluationStatus.InvalidParameters
    with pytest.raises(RuntimeError, match="converged base"):
        sensitivity.steady_base()
    del mesh, sensitivity, solver, restored
    gc.collect()
    np.testing.assert_array_equal(saved["fields"]["U"], fresh["fields"]["U"])
    assert np.isfinite(saved["corrected_fluxes"]).all()
    assert saved["history"].converged
