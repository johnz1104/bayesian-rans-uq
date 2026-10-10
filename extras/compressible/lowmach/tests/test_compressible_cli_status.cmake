# Exercise the real executable and real solver. Status is the acceptance oracle;
# these controls do not certify the CLI's turbulence model or coarse-grid physics.
if(NOT DEFINED CLI OR NOT DEFINED CASE)
    message(FATAL_ERROR "CLI and CASE are required")
endif()

function(check_solve expected_status expected_exit)
    execute_process(COMMAND "${CLI}" ${ARGN}
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(NOT "${result}" STREQUAL "${expected_exit}")
        message(FATAL_ERROR "${expected_status} returned ${result}, expected ${expected_exit}\n${stdout}\n${stderr}")
    endif()
    if(NOT stdout MATCHES "Status: ${expected_status}  ")
        message(FATAL_ERROR "Expected ${expected_status} solver status\n${stdout}\n${stderr}")
    endif()
    message(STATUS "${expected_status} returned ${result}")
endfunction()

if(CASE STREQUAL "capped")
    check_solve(MaxIter 1 --nx 3 --ny 4 --max-iterations 0)
    check_solve(MaxIter 1 --nx 3 --ny 4 --max-iterations 1)
elseif(CASE STREQUAL "diverged")
    # Zero is a supported divergence limit. The positive absolute pressure in
    # the first real iteration must exceed it, independent of residual scaling.
    check_solve(Diverged 1 --nx 3 --ny 4 --max-iterations 1 --divergence-limit 0)
elseif(CASE STREQUAL "converged")
    # Same Ma=0.1 and 1e-3 solver tolerance as the existing smoke fixture.
    # The larger cap allows that existing convergence criterion to be met.
    check_solve(Converged 0 --nx 20 --ny 16 --max-iterations 5000)
elseif(CASE STREQUAL "arguments")
    foreach(arguments IN ITEMS "--unknown" "--nx" "--nx;0" "--ny;-1"
            "--max-iterations;-1" "--max-iterations;1x"
            "--divergence-limit;nan" "--divergence-limit;-1")
        execute_process(COMMAND "${CLI}" ${arguments}
            RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
        if(NOT "${result}" STREQUAL "2" OR NOT stderr MATCHES "Usage:"
                OR stdout MATCHES "Result:")
            message(FATAL_ERROR "Malformed arguments did not fail before solving: ${arguments}\n${result}\n${stdout}\n${stderr}")
        endif()
    endforeach()
else()
    message(FATAL_ERROR "Unknown CASE: ${CASE}")
endif()
