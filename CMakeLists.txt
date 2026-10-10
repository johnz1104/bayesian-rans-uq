cmake_minimum_required(VERSION 3.21)
project(cfdr VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_package(Threads REQUIRED)
option(BUILD_TESTING "Build the C++ test programs (CTest)" ON)
option(CFDR_BUILD_PYTHON "Build the optional cfdr._core Python extension" OFF)
if(BUILD_TESTING)
    enable_testing()
endif()

# Bind the installed binary to the actual source bytes, including local edits.
set(_cfdr_source_globs)
foreach(_dir core rans-uq extras tests/tools)
    list(APPEND _cfdr_source_globs
        "${PROJECT_SOURCE_DIR}/${_dir}/*.cpp" "${PROJECT_SOURCE_DIR}/${_dir}/*.hpp")
endforeach()
file(GLOB_RECURSE _cfdr_sources CONFIGURE_DEPENDS ${_cfdr_source_globs})
list(SORT _cfdr_sources)
set(_cfdr_identity "")
foreach(_file IN LISTS _cfdr_sources)
    file(SHA256 "${_file}" _sha)
    file(RELATIVE_PATH _relative "${PROJECT_SOURCE_DIR}" "${_file}")
    string(APPEND _cfdr_identity "${_relative}:${_sha}\n")
endforeach()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_cfdr_sources})
string(SHA256 CFDR_SOURCE_DIGEST "${_cfdr_identity}")
execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE CFDR_SOURCE_REVISION OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT CFDR_SOURCE_REVISION)
    set(CFDR_SOURCE_REVISION "unknown")
endif()
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" _cfdr_build_type)
set(CFDR_BUILD_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_cfdr_build_type}}")
configure_file(cmake/BuildInfo.hpp.in generated/cfdr/BuildInfo.hpp @ONLY)

include(cmake/Targets.cmake)

configure_package_config_file(cmake/cfdrConfig.cmake.in
    "${PROJECT_BINARY_DIR}/cfdrConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cfdr")
write_basic_package_version_file("${PROJECT_BINARY_DIR}/cfdrConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(EXPORT cfdrTargets NAMESPACE cfdr:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cfdr")
install(FILES "${PROJECT_BINARY_DIR}/cfdrConfig.cmake"
    "${PROJECT_BINARY_DIR}/cfdrConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/cfdr")
install(FILES "${PROJECT_BINARY_DIR}/generated/cfdr/BuildInfo.hpp"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cfdr")
