cmake_minimum_required(VERSION 3.30)

# Set the project name, version, and languages
project(
  volumembo
  VERSION 0.1.0
  LANGUAGES CXX)

# Enable PIC for Python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Create the core volumembo library
add_library(volumembo)

# Add source files and headers to the volumembo library
target_sources(
  volumembo
  PRIVATE lib/median_fitter.cpp
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         include
         FILES
         include/volumembo/median_fitter.hpp
         include/volumembo/priority_queue.hpp
         include/volumembo/pybind11_numpy_interop.hpp
         include/volumembo/span2d.hpp)

# Specify C++20 standard for the core library
target_compile_features(volumembo PUBLIC cxx_std_20)

# Add Python bindings
find_package(pybind11 CONFIG REQUIRED)

# Define Python extension module
pybind11_add_module(_volumembo MODULE src/volumembo/volumembo_python.cpp)
target_link_libraries(_volumembo PUBLIC volumembo)

# Install th Python module library target
install(TARGETS _volumembo DESTINATION .)

# Optionally build tests
include(CTest)
if(BUILD_TESTS)
  add_subdirectory(tests)
endif()
