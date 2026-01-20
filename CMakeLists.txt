cmake_minimum_required(VERSION 3.25)

# ------------------------------------------------
# Set a name and a version number for the project:
# ------------------------------------------------
project(
  volumembo
  VERSION 0.1.0
  LANGUAGES CXX)

# Initialize some default paths
include(GNUInstallDirs)

# Enable PIC for Python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# -------------------
# Compilation options
# -------------------
option(BUILD_PYTHON "Enable building of Python bindings" ON)
option(BUILD_DOCS "Enable building of documentation" OFF)

# ---------------------------
# Add the main library target
# ---------------------------
add_library(volumembo)

# Add source files and headers to the volumembo library
target_sources(volumembo
  PRIVATE
    lib/median_fitter.cpp

  PRIVATE
    FILE_SET private_headers
    TYPE HEADERS
    FILES
      include/volumembo/pybind11_numpy_interop.hpp

  PUBLIC
    FILE_SET public_headers
    TYPE HEADERS
    BASE_DIRS include
    FILES
      include/volumembo/median_fitter.hpp
      include/volumembo/priority_queue.hpp
      include/volumembo/span2d.hpp
)

# Specify C++20 standard for the core library
target_compile_features(volumembo PUBLIC cxx_std_20)

# ----------------------
# Optionally build tests
# ----------------------
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()

# ---------------------
# Optionally build docs
# ---------------------
if(BUILD_DOCS)
  add_subdirectory(doc)
endif()

# ------------------------------
# Optionally build Python module
# ------------------------------
if(BUILD_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)

  # Define Python extension module
  pybind11_add_module(_volumembo MODULE src/volumembo/volumembo_python.cpp)
  target_link_libraries(_volumembo PUBLIC volumembo)

  # Install the Python module library target
  install(TARGETS _volumembo
    LIBRARY
      DESTINATION .
  )
endif()

# ------------------------------
# Install the target and headers
# ------------------------------
install(TARGETS volumembo
    EXPORT volumemboTargets
    FILE_SET public_headers
)

# Install the exported targets (this creates volumemboTargets.cmake)
install(EXPORT volumemboTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/volumembo
  NAMESPACE volumembo::
)

# Include helpers for calling configure_package_config_file
include(CMakePackageConfigHelpers)

# Generate the volumemboConfig.cmake file
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/volumemboConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/volumemboConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/volumembo
)

# Install the generated Config file
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/volumemboConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/volumembo
)

# Generate a version file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/volumemboConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install the version file
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/volumemboConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/volumembo
)
