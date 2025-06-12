cmake_minimum_required(VERSION 3.30)

# Set a name and a version number for your project:
project(
  volumembo
  VERSION 0.1.0
  LANGUAGES CXX)

# Set the C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable PIC for Python bindings
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compilation options
set(BUILD_TESTS
    ON
    CACHE BOOL "Enable building of C++ tests")

# Compile the library
add_library(volumembo lib/median_fitter.cpp)

# Add the include directory
target_include_directories(volumembo PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Add Python bindings
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_volumembo MODULE src/volumembo/volumembo_python.cpp)
target_link_libraries(_volumembo PUBLIC volumembo)
install(TARGETS _volumembo DESTINATION .)

# Compile the tests
if(BUILD_TESTS)
  add_subdirectory(ext/Catch2)
  add_subdirectory(tests)
endif()
