# Run the actual CLI. These checks qualify argument and solve-status behavior,
# not the physical accuracy of the demonstration's empirical correlations.
if(NOT DEFINED CLI OR NOT DEFINED CASE)
    message(FATAL_ERROR "CLI and CASE are required")
endif()

if(CASE STREQUAL "arguments")
    foreach(arguments IN ITEMS "--not-a-mode" "--validate-channel;--unknown"
            "--validate-channel;--max-iterations" "--validate-channel;--max-iterations;-1"
            "--validate-channel;--max-iterations;1x" "--validate-channel;--max-iterations;nan"
            "--validate-channel;--max-iterations;999999999999999999999999"
            "--validate-channel;--max-iterations;1;--max-iterations;2"
            "--validate-channel;--demo" "--help;extra")
        execute_process(COMMAND "${CLI}" ${arguments}
            RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
        if(NOT "${result}" STREQUAL "2" OR NOT stderr MATCHES "Usage:"
                OR stdout MATCHES "Running SIMPLE|Forward Model Demo|Results")
            message(FATAL_ERROR "Malformed arguments did not fail before solving: ${arguments}\n${result}\n${stdout}\n${stderr}")
        endif()
    endforeach()
elseif(CASE STREQUAL "capped")
    foreach(mode IN ITEMS --validate-channel --validate-plate --demo --all)
        execute_process(COMMAND "${CLI}" "${mode}" --max-iterations 0
            RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
        if(NOT "${result}" STREQUAL "1" OR NOT stdout MATCHES "maxIter|Unconverged|MaxIter|NOT CONVERGED"
                OR stdout MATCHES "STATUS: PASS")
            message(FATAL_ERROR "Capped solve claimed success: ${mode}\n${result}\n${stdout}\n${stderr}")
        endif()
    endforeach()
elseif(CASE STREQUAL "observables")
    if(NOT DEFINED ORACLE)
        message(FATAL_ERROR "ORACLE is required for the independent observable comparison")
    endif()
    execute_process(COMMAND "${ORACLE}" --cli-reference
        RESULT_VARIABLE oracle_exit OUTPUT_VARIABLE reference ERROR_VARIABLE reference_error)
    if(NOT "${oracle_exit}" STREQUAL "0")
        message(FATAL_ERROR "Independent reference failed: ${reference_error}")
    endif()
    string(REGEX MATCH "CORE_WALL_CF=([^\n]+)" ignored "${reference}")
    set(expected_wall "${CMAKE_MATCH_1}")
    string(REGEX MATCH "DEAN_CF=([^\n]+)" ignored "${reference}")
    set(expected_dean "${CMAKE_MATCH_1}")
    execute_process(COMMAND "${CLI}" --help OUTPUT_VARIABLE help)
    if(NOT help MATCHES "max-iterations")
        message(FATAL_ERROR "Bounded observable check requires the documented iteration cap")
    endif()
    execute_process(COMMAND "${CLI}" --validate-channel --max-iterations 0
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    string(REGEX MATCH "Cf \\(simulation\\): *([^\n]+)" ignored "${stdout}")
    set(actual_wall "${CMAKE_MATCH_1}")
    string(REGEX MATCH "Cf \\(Dean 1978\\): *([^\n]+)" ignored "${stdout}")
    set(actual_dean "${CMAKE_MATCH_1}")
    if(NOT "${result}" STREQUAL "1" OR NOT "${actual_wall}" STREQUAL "${expected_wall}"
            OR NOT "${actual_dean}" STREQUAL "${expected_dean}")
        message(FATAL_ERROR "CLI diagnostics disagree with core wall stress or full-height Reynolds convention\n${reference}\n${stdout}\n${stderr}")
    endif()
elseif(CASE STREQUAL "help")
    execute_process(COMMAND "${CLI}" --help
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
    if(NOT "${result}" STREQUAL "0" OR NOT stdout MATCHES "Usage:"
            OR NOT stdout MATCHES "max-iterations")
        message(FATAL_ERROR "Help did not document the supported options\n${result}\n${stdout}\n${stderr}")
    endif()
else()
    message(FATAL_ERROR "Unknown CASE: ${CASE}")
endif()
