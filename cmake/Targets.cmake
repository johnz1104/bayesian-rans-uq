# Native library targets, executables, Python bindings and tests. Included from
# the root CMakeLists.txt, so relative paths resolve from the repository root.
# Installed consumers use the cfdr:: targets. Python bindings are opt-in.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- cfdr_core: shared mesh, field, linear-solver and SST infrastructure ----
# Mesh, Field, LinearSolver, BoundaryCondition, SSTModel,
# ObservationOperator (header-only), InferenceParameters (header-only).
add_library(cfdr_core STATIC
    core/src/Mesh.cpp
    core/src/SSTModel.cpp
)
target_include_directories(cfdr_core PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cfdr>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(cfdr_core PUBLIC Threads::Threads)
target_compile_features(cfdr_core PUBLIC cxx_std_17)

# ---- cfdr_incompressible: incompressible SIMPLE solver (rans-uq/) ----
add_library(cfdr_incompressible STATIC
    rans-uq/src/SIMPLESolver.cpp
    rans-uq/src/ForwardModel.cpp
    rans-uq/src/ParameterSensitivity.cpp
)
target_include_directories(cfdr_incompressible PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/rans-uq/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cfdr>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cfdr_incompressible PUBLIC cfdr_core)

add_executable(rans_sst rans-uq/apps/Main.cpp)
target_link_libraries(rans_sst PRIVATE cfdr_incompressible)

# ---- cfdr_compressible: pressure-based low-Mach solver (experimental) ----
add_library(cfdr_compressible STATIC
    extras/compressible/lowmach/src/CompressibleSIMPLESolver.cpp
    extras/compressible/lowmach/src/CompressibleForwardModel.cpp
)
target_include_directories(cfdr_compressible PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/rans-uq/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extras/compressible/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extras/compressible/lowmach/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cfdr>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cfdr_compressible PUBLIC cfdr_incompressible)

add_executable(rans_sst_compressible extras/compressible/lowmach/apps/CompressibleMain.cpp)
target_link_libraries(rans_sst_compressible PRIVATE cfdr_compressible)

# ---- cfdr_dbns: density-based shock-capturing solver ----
add_library(cfdr_dbns STATIC
    extras/compressible/dbns/src/DBNSSolver.cpp
)
target_include_directories(cfdr_dbns PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/core/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extras/compressible/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/extras/compressible/dbns/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/cfdr>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(cfdr_dbns PUBLIC cfdr_core)


# ---- recorder used by the six-case automated golden gate ----
add_executable(record_goldens tests/tools/record_goldens.cpp)
target_link_libraries(record_goldens PRIVATE cfdr_incompressible cfdr_dbns)
target_include_directories(record_goldens PRIVATE extras/compressible/include)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _cfdr_bt)
target_compile_definitions(record_goldens PRIVATE
    CFDR_CXX_FLAGS="${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_cfdr_bt}}")

# Installed consumers use cfdr::<component>; in-tree code may use either name.
foreach(component core incompressible compressible dbns)
    add_library(cfdr::${component} ALIAS cfdr_${component})
    set_target_properties(cfdr_${component} PROPERTIES EXPORT_NAME ${component}
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    install(TARGETS cfdr_${component} EXPORT cfdrTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endforeach()
# Every public header installs flat under include/cfdr/.
foreach(header_dir core/include rans-uq/include extras/compressible/include
                   extras/compressible/lowmach/include extras/compressible/dbns/include)
    install(DIRECTORY ${header_dir}/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cfdr FILES_MATCHING PATTERN "*.hpp")
endforeach()

if(CFDR_BUILD_PYTHON)
    find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _pybind11_dir OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pybind11_status)
    if(NOT _pybind11_status EQUAL 0)
        message(FATAL_ERROR "CFDR_BUILD_PYTHON requires pybind11 in ${Python_EXECUTABLE}")
    endif()
    find_package(pybind11 CONFIG REQUIRED HINTS "${_pybind11_dir}")
    pybind11_add_module(_core NO_EXTRAS rans-uq/python/bindings.cpp)
    target_link_libraries(_core PRIVATE cfdr::incompressible cfdr::compressible cfdr::dbns)
    set(CFDR_PYTHON_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/python/cfdr" CACHE PATH
        "Build-tree Python package directory")
    set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${CFDR_PYTHON_OUTPUT_DIRECTORY}")
    file(GLOB_RECURSE _cfdr_python_sources CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/rans-uq/python/cfdr/*.py"
        "${PROJECT_SOURCE_DIR}/rans-uq/python/cfdr/*.pyi")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_cfdr_python_sources})
    file(COPY "${PROJECT_SOURCE_DIR}/rans-uq/python/cfdr/" DESTINATION "${CFDR_PYTHON_OUTPUT_DIRECTORY}")
    set(CFDR_PYTHON_INSTALL_DIRECTORY
        "${CMAKE_INSTALL_LIBDIR}/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages/cfdr"
        CACHE STRING "Python package install directory relative to prefix")
    install(TARGETS _core LIBRARY DESTINATION "${CFDR_PYTHON_INSTALL_DIRECTORY}")
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/rans-uq/python/cfdr/"
        DESTINATION "${CFDR_PYTHON_INSTALL_DIRECTORY}"
        FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi" PATTERN "py.typed")
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
