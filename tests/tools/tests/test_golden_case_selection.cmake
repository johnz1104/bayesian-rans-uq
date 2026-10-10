# Exercise the actual recorder CLI; a partially matching selection must fail
# before it runs a solver or creates any golden file.
if(NOT DEFINED RECORDER OR NOT DEFINED WORK_DIRECTORY)
    message(FATAL_ERROR "RECORDER and WORK_DIRECTORY are required")
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef run_id)
set(output "${WORK_DIRECTORY}/golden-case-selection-${run_id}")
file(MAKE_DIRECTORY "${output}")
execute_process(
    COMMAND "${RECORDER}" "${output}" synthetic-case-selection 0
            dbns_shock_tubes unknown_case_regression
    RESULT_VARIABLE rejected OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
file(GLOB unexpected "${output}/*.json")
file(REMOVE_RECURSE "${output}")
if(rejected EQUAL 0 OR unexpected OR stdout MATCHES "case dbns_shock_tubes")
    message(FATAL_ERROR "Mixed invalid selection ran or succeeded: exit=${rejected}\n${stdout}\n${stderr}")
endif()
if(NOT stderr MATCHES "unknown_case_regression")
    message(FATAL_ERROR "Failure did not identify the unknown case: ${stderr}")
endif()

file(MAKE_DIRECTORY "${output}")
execute_process(
    COMMAND "${RECORDER}" "${output}" synthetic-case-selection 0 dbns_shock_tubes
    RESULT_VARIABLE accepted OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr)
set(valid_file "${output}/dbns_shock_tubes.json")
if(NOT accepted EQUAL 0 OR NOT EXISTS "${valid_file}")
    file(REMOVE_RECURSE "${output}")
    message(FATAL_ERROR "Supported selection did not record: ${accepted}\n${stdout}\n${stderr}")
endif()
file(READ "${valid_file}" document)
string(JSON case_name GET "${document}" case)
file(REMOVE_RECURSE "${output}")
if(NOT case_name STREQUAL "dbns_shock_tubes")
    message(FATAL_ERROR "Supported case wrote the wrong record: ${case_name}")
endif()
message(STATUS "Invalid selection rejected before execution; supported case recorded")
