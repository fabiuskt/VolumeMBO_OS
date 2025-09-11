# ------------------------------------------------------------------------------
# Enable CMake's FetchContent module, which allows downloading dependencies like
# Catch2 automatically at configure time.
# ------------------------------------------------------------------------------
include(FetchContent)

# ------------------------------------------------------------------------------
# Declare a dependency on Catch2 via FetchContent. This will clone the Catch2
# GitHub repository during configuration.
# ------------------------------------------------------------------------------
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.10.0)

# ------------------------------------------------------------------------------
# Make Catch2 available in this project. This defines the CMake targets like
# Catch2::Catch2 and Catch2::Catch2WithMain.
# ------------------------------------------------------------------------------
FetchContent_MakeAvailable(Catch2)

# ------------------------------------------------------------------------------
# Make Catch2 available in this project. This defines the CMake targets like
# Catch2::Catch2 and Catch2::Catch2WithMain.
# ------------------------------------------------------------------------------
include(Catch)

# ------------------------------------------------------------------------------
# Create a single test binary that compiles all unit test files. Each file
# should use TEST_CASE(...) macros from Catch2.
# ------------------------------------------------------------------------------
add_executable(tests priority_queue_t.cpp median_fitter_t.cpp)

# Link the test binary against: - the main library under test - the test helper
# library ( - the Catch2 main entry point (Catch2::Catch2WithMain)
target_link_libraries(tests PUBLIC volumembo Catch2::Catch2WithMain)

# ------------------------------------------------------------------------------
# Register the tests with CTest so that `ctest` or `make test` will run them.
# This parses all TEST_CASE(...) macros in the 'tests' binary.
# ------------------------------------------------------------------------------
catch_discover_tests(tests)
