# Native verification and regression tests registered with CTest.
#
# Test sources live beside the component they exercise; tests that span
# components link every target they use.
#
# Conventions:
#   - Every test executable returns 0 on success, non-zero on failure.
#   - Tests are deterministic (fixed seeds; no wall-clock dependence).
#   - Long-running tests carry the "slow" label; tests/tools/check.sh skips them
#     unless called with --all.

set(CORE_TESTS ${PROJECT_SOURCE_DIR}/core/tests)
set(RANS_TESTS ${PROJECT_SOURCE_DIR}/rans-uq/tests)
set(DBNS_TESTS ${PROJECT_SOURCE_DIR}/extras/compressible/dbns/tests)
set(LOWMACH_TESTS ${PROJECT_SOURCE_DIR}/extras/compressible/lowmach/tests)
set(TOOLS_TESTS ${PROJECT_SOURCE_DIR}/tests/tools/tests)

add_executable(test_core_math_contracts ${CORE_TESTS}/test_core_math_contracts.cpp)
target_link_libraries(test_core_math_contracts PRIVATE cfdr_incompressible cfdr_dbns)
foreach(contract trig jacobi simple dbns amg)
    add_test(NAME core_math_${contract} COMMAND test_core_math_contracts ${contract})
    set_tests_properties(core_math_${contract} PROPERTIES TIMEOUT 30 LABELS "unit;math")
endforeach()

add_executable(test_riemann_reference_contract ${DBNS_TESTS}/test_riemann_reference_contract.cpp)
target_link_libraries(test_riemann_reference_contract PRIVATE cfdr_dbns)
foreach(contract invalid vacuum unresolved)
    add_test(NAME riemann_reference_${contract} COMMAND test_riemann_reference_contract ${contract})
    set_tests_properties(riemann_reference_${contract} PROPERTIES TIMEOUT 30 LABELS "unit;math;dbns")
endforeach()

# ---- mesh_bfs: BFS mesh factory smoke test ----
add_executable(test_mesh_bfs ${CORE_TESTS}/test_mesh_bfs.cpp)
target_link_libraries(test_mesh_bfs PRIVATE cfdr_core)
add_test(NAME mesh_bfs COMMAND test_mesh_bfs)
set_tests_properties(mesh_bfs PROPERTIES TIMEOUT 30 LABELS "unit;mesh")

# ---- couette_bc: moving-wall (Couette) boundary-condition test ----
add_executable(test_couette_bc ${CORE_TESTS}/test_couette_bc.cpp)
target_link_libraries(test_couette_bc PRIVATE cfdr_core)
add_test(NAME couette_bc COMMAND test_couette_bc)
set_tests_properties(couette_bc PROPERTIES TIMEOUT 30 LABELS "unit;bc")

# ---- bfs_lemoin_bc: free-slip top + profile inflow (Le-Moin BFS) BC test ----
add_executable(test_bfs_lemoin_bc ${CORE_TESTS}/test_bfs_lemoin_bc.cpp)
target_link_libraries(test_bfs_lemoin_bc PRIVATE cfdr_core)
add_test(NAME bfs_lemoin_bc COMMAND test_bfs_lemoin_bc)
set_tests_properties(bfs_lemoin_bc PROPERTIES TIMEOUT 30 LABELS "unit;bc")

# ---- transpose_stress: Boussinesq transpose-stress deferred correction ----
add_executable(test_transpose_stress ${CORE_TESTS}/test_transpose_stress.cpp)
target_link_libraries(test_transpose_stress PRIVATE cfdr_core)
add_test(NAME transpose_stress COMMAND test_transpose_stress)
set_tests_properties(transpose_stress PROPERTIES TIMEOUT 30 LABELS "unit;solver")

# ---- sst_production_2003: corrected omega production (limiter-discriminating) ----
add_executable(test_sst_production_2003 ${CORE_TESTS}/test_sst_production_2003.cpp)
target_link_libraries(test_sst_production_2003 PRIVATE cfdr_core)
add_test(NAME sst_production_2003 COMMAND test_sst_production_2003)
set_tests_properties(sst_production_2003 PROPERTIES TIMEOUT 30 LABELS "unit;sst")

add_executable(test_sst_strain_output ${CORE_TESTS}/test_sst_strain_output.cpp)
target_link_libraries(test_sst_strain_output PRIVATE cfdr::core)
add_test(NAME sst_strain_output COMMAND test_sst_strain_output)
set_tests_properties(sst_strain_output PROPERTIES TIMEOUT 30 LABELS "unit;sst")

# ---- cp_normalization: pressure-coefficient reference invariance ----
add_executable(test_cp_normalization ${CORE_TESTS}/test_cp_normalization.cpp)
target_link_libraries(test_cp_normalization PRIVATE cfdr_core)
add_test(NAME cp_normalization COMMAND test_cp_normalization)
set_tests_properties(cp_normalization PROPERTIES TIMEOUT 30 LABELS "unit;observation")

# ---- infra_fixes: Gauss-Seidel sweep, inBounds guard, cache copy, RC probe ----
add_executable(test_infra_fixes ${RANS_TESTS}/test_infra_fixes.cpp)
target_link_libraries(test_infra_fixes PRIVATE cfdr_incompressible)
add_test(NAME infra_fixes COMMAND test_infra_fixes)
set_tests_properties(infra_fixes PROPERTIES TIMEOUT 240 LABELS "unit;infra")

# ---- injection_source: Reynolds-stress injection manufactured-field test ----
add_executable(test_injection_source ${RANS_TESTS}/test_injection_source.cpp)
target_link_libraries(test_injection_source PRIVATE cfdr_incompressible)
add_test(NAME injection_source COMMAND test_injection_source)
set_tests_properties(injection_source PROPERTIES TIMEOUT 30 LABELS "unit;injection")

# ---- nut_floor_release: startup-only floor + molecular wall-stress observable ----
add_executable(test_nut_floor_release ${RANS_TESTS}/test_nut_floor_release.cpp)
target_link_libraries(test_nut_floor_release PRIVATE cfdr_incompressible)
add_test(NAME nut_floor_release COMMAND test_nut_floor_release)
set_tests_properties(nut_floor_release PROPERTIES TIMEOUT 240 LABELS "unit;solver;sst")

# ---- convergence_accounting: carried turbulence norms under update intervals ----
add_executable(test_convergence_accounting ${RANS_TESTS}/test_convergence_accounting.cpp)
target_link_libraries(test_convergence_accounting PRIVATE cfdr_incompressible)
add_test(NAME convergence_accounting COMMAND test_convergence_accounting)

add_executable(test_state_validation ${LOWMACH_TESTS}/test_state_validation.cpp)
target_link_libraries(test_state_validation PRIVATE cfdr_compressible)
add_test(NAME state_validation COMMAND test_state_validation)
set_tests_properties(convergence_accounting PROPERTIES TIMEOUT 240 LABELS "unit;solver")

# ---- periodic_channel: streamwise-periodic curved-channel mesh + body force ----
add_executable(test_periodic_channel ${RANS_TESTS}/test_periodic_channel.cpp)
target_link_libraries(test_periodic_channel PRIVATE cfdr_incompressible)
add_test(NAME periodic_channel COMMAND test_periodic_channel)
set_tests_properties(periodic_channel PROPERTIES TIMEOUT 240 LABELS "unit;mesh;solver")

# ---- compressible_smoke: Ma=0.1 channel smoke convergence ----
add_executable(test_compressible_smoke ${LOWMACH_TESTS}/test_compressible_smoke.cpp)
target_link_libraries(test_compressible_smoke PRIVATE cfdr_compressible)
add_test(NAME compressible_smoke COMMAND test_compressible_smoke)

foreach(status capped diverged converged arguments)
    add_test(NAME compressible_cli_${status}
        COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:rans_sst_compressible>
            -DCASE=${status} -P ${LOWMACH_TESTS}/test_compressible_cli_status.cmake)
    set_tests_properties(compressible_cli_${status} PROPERTIES
        TIMEOUT 120 LABELS "unit;compressible;cli;safety")
endforeach()

add_executable(test_two_pressure_contract ${LOWMACH_TESTS}/test_two_pressure_contract.cpp)
target_link_libraries(test_two_pressure_contract PRIVATE cfdr_compressible)
add_test(NAME two_pressure_contract COMMAND test_two_pressure_contract)

add_executable(test_compressible_observation_contract
               ${LOWMACH_TESTS}/test_compressible_observation_contract.cpp)
target_link_libraries(test_compressible_observation_contract PRIVATE cfdr_compressible)
add_test(NAME compressible_observation_contract
         COMMAND test_compressible_observation_contract)
set_tests_properties(compressible_observation_contract PROPERTIES
    TIMEOUT 30
    LABELS "unit;observation;compressible")
set_tests_properties(compressible_smoke PROPERTIES
    TIMEOUT 180
    LABELS "smoke;compressible")

# ---- dbns_numerics: density-based solver foundational numerics (header-only) ----
add_executable(test_dbns_numerics ${DBNS_TESTS}/test_dbns_numerics.cpp)
target_include_directories(test_dbns_numerics PRIVATE
    ${PROJECT_SOURCE_DIR}/extras/compressible/dbns/include)
target_link_libraries(test_dbns_numerics PRIVATE cfdr_compressible)
add_test(NAME dbns_numerics COMMAND test_dbns_numerics)
set_tests_properties(dbns_numerics PROPERTIES TIMEOUT 30 LABELS "unit;dbns")

# ---- dbns_verification: shock tube vs exact Riemann + MMS order of accuracy ----
add_executable(test_dbns_verification ${DBNS_TESTS}/test_dbns_verification.cpp)
target_link_libraries(test_dbns_verification PRIVATE cfdr_dbns)
add_test(NAME dbns_verification COMMAND test_dbns_verification)
set_tests_properties(dbns_verification PROPERTIES TIMEOUT 240 LABELS "verification;dbns")

# ---- dbns_verification2: oblique-shock reflection + wall observation operator ----
add_executable(test_dbns_verification2 ${DBNS_TESTS}/test_dbns_verification2.cpp)
target_link_libraries(test_dbns_verification2 PRIVATE cfdr_dbns)
add_test(NAME dbns_verification2 COMMAND test_dbns_verification2)
set_tests_properties(dbns_verification2 PROPERTIES TIMEOUT 120 LABELS "verification;dbns")

# ---- dbns_implicit: LU-SGS viscous-steady tests (Couette exact + laminar plate) ----
add_executable(test_dbns_implicit ${DBNS_TESTS}/test_dbns_implicit.cpp)
target_link_libraries(test_dbns_implicit PRIVATE cfdr_dbns)
add_test(NAME dbns_implicit COMMAND test_dbns_implicit)
set_tests_properties(dbns_implicit PROPERTIES TIMEOUT 480 LABELS "verification;dbns")

# ---- dbns_injection: deferred-correction coupling with the heat-flux reach ----
add_executable(test_dbns_injection ${DBNS_TESTS}/test_dbns_injection.cpp)
target_link_libraries(test_dbns_injection PRIVATE cfdr_dbns)
add_test(NAME dbns_injection COMMAND test_dbns_injection)
set_tests_properties(dbns_injection PROPERTIES TIMEOUT 480 LABELS "verification;dbns")

# ---- dbns_sst_wall: omega wall anchoring sustains an imposed turbulent layer ----
add_executable(test_dbns_sst_wall ${DBNS_TESTS}/test_dbns_sst_wall.cpp)
target_link_libraries(test_dbns_sst_wall PRIVATE cfdr_dbns)
add_test(NAME dbns_sst_wall COMMAND test_dbns_sst_wall)
set_tests_properties(dbns_sst_wall PROPERTIES TIMEOUT 600 LABELS "verification;dbns")

# ---- mesh_volume_convention: one cell-geometry convention on every mesh path ----
add_executable(test_mesh_volume_convention ${CORE_TESTS}/test_mesh_volume_convention.cpp)
target_link_libraries(test_mesh_volume_convention PRIVATE cfdr_core)
add_test(NAME mesh_volume_convention COMMAND test_mesh_volume_convention)
set_tests_properties(mesh_volume_convention PROPERTIES TIMEOUT 30 LABELS "unit;mesh")

# ---- convergence_criteria: declared convergence obeys exact discrete identities ----
add_executable(test_convergence_criteria ${RANS_TESTS}/test_convergence_criteria.cpp)
target_link_libraries(test_convergence_criteria PRIVATE cfdr_incompressible)
add_test(NAME convergence_criteria COMMAND test_convergence_criteria)
set_tests_properties(convergence_criteria PROPERTIES TIMEOUT 900 LABELS "unit;solver")

# ---- fine_mesh_convergence: Couette ny=384 must not stop on a transient (slow) ----
add_executable(test_fine_mesh_convergence ${RANS_TESTS}/test_fine_mesh_convergence.cpp)
target_link_libraries(test_fine_mesh_convergence PRIVATE cfdr_incompressible)
add_test(NAME fine_mesh_convergence COMMAND test_fine_mesh_convergence)
set_tests_properties(fine_mesh_convergence PROPERTIES TIMEOUT 7200 LABELS "slow;solver")

# ---- amg_reuse: frozen-hierarchy refreshes and a symmetric V-cycle ----
add_executable(test_amg_reuse ${CORE_TESTS}/test_amg_reuse.cpp)
target_link_libraries(test_amg_reuse PRIVATE cfdr_core)
add_test(NAME amg_reuse COMMAND test_amg_reuse)
set_tests_properties(amg_reuse PROPERTIES TIMEOUT 60 LABELS "unit;linalg")

# ---- probe_interpolation: probes interpolate instead of snapping to a cell ----
add_executable(test_probe_interpolation ${CORE_TESTS}/test_probe_interpolation.cpp)
target_link_libraries(test_probe_interpolation PRIVATE cfdr_core)
add_test(NAME probe_interpolation COMMAND test_probe_interpolation)
set_tests_properties(probe_interpolation PROPERTIES TIMEOUT 60 LABELS "unit;observation")

# ---- rhie_chow_policy: Rhie-Chow on every mesh, alpha_U-independent, stored fluxes ----
add_executable(test_rhie_chow_policy ${RANS_TESTS}/test_rhie_chow_policy.cpp)
target_link_libraries(test_rhie_chow_policy PRIVATE cfdr_incompressible)
add_test(NAME rhie_chow_policy COMMAND test_rhie_chow_policy)
set_tests_properties(rhie_chow_policy PROPERTIES TIMEOUT 1200 LABELS "unit;solver")

# ---- periodic_channel_retau: canonical periodic channel at a prescribed Re_tau ----
add_executable(test_periodic_channel_retau ${RANS_TESTS}/test_periodic_channel_retau.cpp)
target_link_libraries(test_periodic_channel_retau PRIVATE cfdr_incompressible)
add_test(NAME periodic_channel_retau COMMAND test_periodic_channel_retau)
set_tests_properties(periodic_channel_retau PROPERTIES TIMEOUT 900 LABELS "unit;solver;mesh")

# ---- convection_mms: observed order of the SIMPLE convection scheme ----
add_executable(test_convection_mms ${RANS_TESTS}/test_convection_mms.cpp)
target_link_libraries(test_convection_mms PRIVATE cfdr_incompressible)
add_test(NAME convection_mms COMMAND test_convection_mms)
set_tests_properties(convection_mms PROPERTIES TIMEOUT 1800 LABELS "verification;solver")

# Stable native extension points used by downstream research consumers.
add_executable(test_research_interfaces ${RANS_TESTS}/test_research_interfaces.cpp)
target_link_libraries(test_research_interfaces PRIVATE cfdr::incompressible)
add_test(NAME research_interfaces COMMAND test_research_interfaces)
set_tests_properties(research_interfaces PROPERTIES TIMEOUT 30 LABELS "unit;research")

add_executable(test_evaluation_safety ${RANS_TESTS}/test_evaluation_safety.cpp)
target_link_libraries(test_evaluation_safety PRIVATE cfdr::incompressible)
add_test(NAME evaluation_safety COMMAND test_evaluation_safety)
set_tests_properties(evaluation_safety PROPERTIES TIMEOUT 60 LABELS "unit;safety")

add_executable(test_krylov_residual ${CORE_TESTS}/test_krylov_residual.cpp)
target_link_libraries(test_krylov_residual PRIVATE cfdr::core)
add_test(NAME krylov_residual COMMAND test_krylov_residual)
set_tests_properties(krylov_residual PROPERTIES TIMEOUT 30 LABELS "unit;linalg")

add_executable(test_diagnostic_partials ${RANS_TESTS}/test_diagnostic_partials.cpp)
target_link_libraries(test_diagnostic_partials PRIVATE cfdr::incompressible)
add_test(NAME diagnostic_partials COMMAND test_diagnostic_partials)
set_tests_properties(diagnostic_partials PROPERTIES TIMEOUT 60 LABELS "unit;sensitivity")

add_executable(test_steady_operator ${RANS_TESTS}/test_steady_operator.cpp)
target_link_libraries(test_steady_operator PRIVATE cfdr::incompressible)
add_test(NAME steady_operator COMMAND test_steady_operator)
set_tests_properties(steady_operator PROPERTIES TIMEOUT 60 LABELS "unit;sensitivity")

add_executable(test_steady_sensitivity ${RANS_TESTS}/test_steady_sensitivity.cpp)
target_link_libraries(test_steady_sensitivity PRIVATE cfdr::incompressible)
add_test(NAME steady_sensitivity COMMAND test_steady_sensitivity)
set_tests_properties(steady_sensitivity PROPERTIES TIMEOUT 120 LABELS "unit;sensitivity")

add_executable(test_warm_fd_refinement ${RANS_TESTS}/test_warm_fd_refinement.cpp)
target_link_libraries(test_warm_fd_refinement PRIVATE cfdr::incompressible)
add_test(NAME warm_fd_refinement COMMAND test_warm_fd_refinement)
set_tests_properties(warm_fd_refinement PROPERTIES TIMEOUT 180 LABELS "unit;sensitivity")

add_executable(test_linear_scratch ${CORE_TESTS}/test_linear_scratch.cpp)
target_link_libraries(test_linear_scratch PRIVATE cfdr::core)
add_test(NAME linear_scratch COMMAND test_linear_scratch)
set_tests_properties(linear_scratch PROPERTIES TIMEOUT 30 LABELS "unit;linalg")

add_executable(test_mesh_validity ${RANS_TESTS}/test_mesh_validity.cpp)
target_link_libraries(test_mesh_validity PRIVATE cfdr::incompressible cfdr::dbns)
add_test(NAME mesh_validity COMMAND test_mesh_validity)
set_tests_properties(mesh_validity PROPERTIES TIMEOUT 30 LABELS "unit;mesh")

add_executable(test_prepared_observations ${CORE_TESTS}/test_prepared_observations.cpp)
target_link_libraries(test_prepared_observations PRIVATE cfdr::core)
add_test(NAME prepared_observations COMMAND test_prepared_observations)
set_tests_properties(prepared_observations PROPERTIES TIMEOUT 30 LABELS "unit;observation")

add_executable(test_gradient_workspace ${RANS_TESTS}/test_gradient_workspace.cpp)
target_link_libraries(test_gradient_workspace PRIVATE cfdr::core)
add_test(NAME gradient_workspace COMMAND test_gradient_workspace)
set_tests_properties(gradient_workspace PROPERTIES TIMEOUT 30 LABELS "unit;sst")

add_executable(test_residual_workspace ${RANS_TESTS}/test_residual_workspace.cpp)
target_link_libraries(test_residual_workspace PRIVATE cfdr::incompressible)
add_test(NAME residual_workspace COMMAND test_residual_workspace)
set_tests_properties(residual_workspace PROPERTIES TIMEOUT 30 LABELS "unit;sensitivity")

add_executable(test_forward_workspace ${RANS_TESTS}/test_forward_workspace.cpp)
target_link_libraries(test_forward_workspace PRIVATE cfdr::incompressible)
add_test(NAME forward_workspace COMMAND test_forward_workspace)
set_tests_properties(forward_workspace PROPERTIES TIMEOUT 30 LABELS "unit;workspace")

add_executable(test_forward_batch ${RANS_TESTS}/test_forward_batch.cpp)
target_link_libraries(test_forward_batch PRIVATE cfdr::incompressible)
add_test(NAME forward_batch COMMAND test_forward_batch)
set_tests_properties(forward_batch PROPERTIES TIMEOUT 60 LABELS "unit;batch")

add_executable(test_dbns_math_contracts ${DBNS_TESTS}/test_dbns_math_contracts.cpp)
target_link_libraries(test_dbns_math_contracts PRIVATE cfdr::dbns)
foreach(contract capped steady wall_work inviscid_wall passive_energy rk_properties periodic endpoint nonfinite
                 laminar_energy)
    add_test(NAME dbns_math_${contract} COMMAND test_dbns_math_contracts ${contract})
    set_tests_properties(dbns_math_${contract} PROPERTIES TIMEOUT 60 LABELS "verification;dbns;math")
endforeach()

add_executable(test_dbns_field_size ${DBNS_TESTS}/test_dbns_field_size.cpp)
target_link_libraries(test_dbns_field_size PRIVATE cfdr::dbns)
foreach(contract short long)
    add_test(NAME dbns_field_size_${contract} COMMAND test_dbns_field_size ${contract})
    set_tests_properties(dbns_field_size_${contract} PROPERTIES TIMEOUT 30 LABELS "unit;dbns;safety")
endforeach()
add_executable(test_likelihood_domain ${RANS_TESTS}/test_likelihood_domain.cpp)
target_link_libraries(test_likelihood_domain PRIVATE cfdr::incompressible)
add_test(NAME likelihood_domain COMMAND test_likelihood_domain)
set_tests_properties(likelihood_domain PROPERTIES TIMEOUT 30 LABELS "unit;observation;safety")

add_executable(test_compressible_likelihood_admission ${LOWMACH_TESTS}/test_compressible_likelihood_admission.cpp)
target_link_libraries(test_compressible_likelihood_admission PRIVATE cfdr::compressible)
add_test(NAME compressible_likelihood_admission COMMAND test_compressible_likelihood_admission)
set_tests_properties(compressible_likelihood_admission PROPERTIES TIMEOUT 30 LABELS "unit;compressible;safety")

add_executable(test_likelihood_small_variance ${RANS_TESTS}/test_likelihood_small_variance.cpp)
target_link_libraries(test_likelihood_small_variance PRIVATE cfdr::core)
add_test(NAME likelihood_small_variance COMMAND test_likelihood_small_variance)
set_tests_properties(likelihood_small_variance PROPERTIES TIMEOUT 30 LABELS "unit;observation;safety")

add_executable(test_compressible_retained_state ${LOWMACH_TESTS}/test_compressible_retained_state.cpp)
target_link_libraries(test_compressible_retained_state PRIVATE cfdr::compressible)
add_test(NAME compressible_retained_state COMMAND test_compressible_retained_state)
set_tests_properties(compressible_retained_state PROPERTIES TIMEOUT 30 LABELS "unit;compressible;safety")

# Sweep regressions: these test native admission before solver state can be used.
add_executable(test_convergence_monitor_validation ${RANS_TESTS}/test_convergence_monitor_validation.cpp)
target_link_libraries(test_convergence_monitor_validation PRIVATE cfdr::incompressible)
add_test(NAME convergence_monitor_validation COMMAND test_convergence_monitor_validation)
set_tests_properties(convergence_monitor_validation PROPERTIES TIMEOUT 60 LABELS "unit;safety;solver")

add_executable(test_solver_settings_validation ${RANS_TESTS}/test_solver_settings_validation.cpp)
target_link_libraries(test_solver_settings_validation PRIVATE cfdr::compressible)
add_test(NAME solver_settings_validation COMMAND test_solver_settings_validation)
set_tests_properties(solver_settings_validation PROPERTIES TIMEOUT 30 LABELS "unit;safety;solver")

add_executable(test_boundary_profile_validation ${CORE_TESTS}/test_boundary_profile_validation.cpp)
target_link_libraries(test_boundary_profile_validation PRIVATE cfdr::core)
foreach(field velocity k omega)
    foreach(layout valid short long patches)
        add_test(NAME boundary_profile_${field}_${layout}
                 COMMAND test_boundary_profile_validation ${field} ${layout})
        set_tests_properties(boundary_profile_${field}_${layout} PROPERTIES
                             TIMEOUT 30 LABELS "unit;safety;bc")
    endforeach()
endforeach()

add_executable(test_linear_solver_selection ${RANS_TESTS}/test_linear_solver_selection.cpp)
target_link_libraries(test_linear_solver_selection PRIVATE cfdr::core)
add_test(NAME linear_solver_selection COMMAND test_linear_solver_selection)
set_tests_properties(linear_solver_selection PROPERTIES TIMEOUT 30 LABELS "unit;linalg;safety")
add_executable(test_sst_variant_validation ${RANS_TESTS}/test_sst_variant_validation.cpp)
target_link_libraries(test_sst_variant_validation PRIVATE cfdr::compressible)
add_test(NAME sst_variant_validation COMMAND test_sst_variant_validation)
set_tests_properties(sst_variant_validation PROPERTIES TIMEOUT 30 LABELS "unit;sst;safety")

add_executable(test_compressible_sst_selection ${LOWMACH_TESTS}/test_compressible_sst_selection.cpp)
target_link_libraries(test_compressible_sst_selection PRIVATE cfdr::compressible)
add_test(NAME compressible_sst_selection COMMAND test_compressible_sst_selection)
set_tests_properties(compressible_sst_selection PROPERTIES TIMEOUT 30 LABELS "unit;sst;compressible")

add_executable(test_zero_convergence_tolerance ${RANS_TESTS}/test_zero_convergence_tolerance.cpp)
target_link_libraries(test_zero_convergence_tolerance PRIVATE cfdr::compressible)
add_test(NAME zero_convergence_tolerance COMMAND test_zero_convergence_tolerance)
set_tests_properties(zero_convergence_tolerance PROPERTIES TIMEOUT 30 LABELS "unit;safety;solver")

add_executable(test_golden_writer_io ${TOOLS_TESTS}/test_golden_writer_io.cpp)
target_include_directories(test_golden_writer_io PRIVATE ${PROJECT_SOURCE_DIR}/tests/tools)
add_test(NAME golden_writer_io COMMAND test_golden_writer_io)
set_tests_properties(golden_writer_io PROPERTIES TIMEOUT 30 LABELS "unit;tooling;safety")

add_executable(test_golden_writer_numbers ${TOOLS_TESTS}/test_golden_writer_numbers.cpp)
target_include_directories(test_golden_writer_numbers PRIVATE ${PROJECT_SOURCE_DIR}/tests/tools)
add_test(NAME golden_writer_numbers COMMAND test_golden_writer_numbers)
set_tests_properties(golden_writer_numbers PROPERTIES TIMEOUT 30 LABELS "unit;tooling;safety")

add_test(NAME golden_case_selection
    COMMAND ${CMAKE_COMMAND} -DRECORDER=$<TARGET_FILE:record_goldens>
        -DWORK_DIRECTORY=${CMAKE_CURRENT_BINARY_DIR}
        -P ${TOOLS_TESTS}/test_golden_case_selection.cmake)
set_tests_properties(golden_case_selection PROPERTIES TIMEOUT 30 LABELS "unit;tooling;safety")

add_executable(test_wall_omega_contracts ${RANS_TESTS}/test_wall_omega_contracts.cpp)
target_link_libraries(test_wall_omega_contracts PRIVATE cfdr::incompressible)
add_test(NAME wall_omega_contracts COMMAND test_wall_omega_contracts)
set_tests_properties(wall_omega_contracts PROPERTIES TIMEOUT 30 LABELS "unit;solver;verification")

add_executable(test_simple_config ${RANS_TESTS}/test_simple_config.cpp)
target_link_libraries(test_simple_config PRIVATE cfdr::incompressible)
add_test(NAME simple_config COMMAND test_simple_config)
set_tests_properties(simple_config PROPERTIES TIMEOUT 30 LABELS "unit;solver;safety")

add_executable(test_batch_determinism ${RANS_TESTS}/test_batch_determinism.cpp)
target_link_libraries(test_batch_determinism PRIVATE cfdr::incompressible)
add_test(NAME batch_determinism COMMAND test_batch_determinism)
set_tests_properties(batch_determinism PROPERTIES TIMEOUT 120 LABELS "unit;solver;safety")

add_executable(test_sst_channel_grid_convergence ${RANS_TESTS}/test_sst_channel_grid_convergence.cpp)
target_link_libraries(test_sst_channel_grid_convergence PRIVATE cfdr::incompressible)
add_test(NAME sst_channel_grid_convergence COMMAND test_sst_channel_grid_convergence)
set_tests_properties(sst_channel_grid_convergence PROPERTIES TIMEOUT 600 LABELS "slow;solver;verification")

# Instrumentation can make unchanged numerical tests much slower. Ordinary
# builds retain their original deadlines, and this knob cannot alter a solver
# tolerance, iteration cap or expected value.
set(CFDR_SANITIZER_TIMEOUT_MULTIPLIER "1" CACHE STRING
    "Positive integer multiplier for explicit test timeouts in sanitizer builds")
if(NOT CFDR_SANITIZER_TIMEOUT_MULTIPLIER MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "CFDR_SANITIZER_TIMEOUT_MULTIPLIER must be a positive integer")
endif()
if(NOT CFDR_SANITIZER_TIMEOUT_MULTIPLIER STREQUAL "1")
    if(NOT CMAKE_CXX_FLAGS MATCHES "-fsanitize=")
        message(FATAL_ERROR
            "CFDR_SANITIZER_TIMEOUT_MULTIPLIER requires -fsanitize= in CMAKE_CXX_FLAGS")
    endif()
    get_property(_cfdr_tests DIRECTORY PROPERTY TESTS)
    foreach(_cfdr_test IN LISTS _cfdr_tests)
        get_property(_cfdr_has_timeout TEST "${_cfdr_test}" PROPERTY TIMEOUT SET)
        if(_cfdr_has_timeout)
            get_property(_cfdr_timeout TEST "${_cfdr_test}" PROPERTY TIMEOUT)
            math(EXPR _cfdr_instrumented_timeout
                "${_cfdr_timeout} * ${CFDR_SANITIZER_TIMEOUT_MULTIPLIER}")
            set_tests_properties("${_cfdr_test}" PROPERTIES TIMEOUT "${_cfdr_instrumented_timeout}")
        endif()
    endforeach()
endif()

add_executable(test_simple_convergence ${RANS_TESTS}/test_simple_convergence.cpp)
target_link_libraries(test_simple_convergence PRIVATE cfdr::incompressible)
add_test(NAME simple_convergence COMMAND test_simple_convergence)
set_tests_properties(simple_convergence PROPERTIES TIMEOUT 120 LABELS "solver;safety")

foreach(status arguments capped help observables)
    add_test(NAME simple_cli_${status}
        COMMAND ${CMAKE_COMMAND} -DCLI=$<TARGET_FILE:rans_sst>
            -DORACLE=$<TARGET_FILE:test_simple_convergence>
            -DCASE=${status} -P ${RANS_TESTS}/test_simple_cli_status.cmake)
    set_tests_properties(simple_cli_${status} PROPERTIES TIMEOUT 60 LABELS "unit;solver;cli;safety")
endforeach()
