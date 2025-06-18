# The Catch test executable
add_executable(tests priority_queue_t.cpp median_fitter_t.cpp)
target_link_libraries(tests PUBLIC volumembo Catch2::Catch2WithMain)

# Get the Catch2 test discovery macro (for Catch2 v3)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../ext/Catch2/extras/Catch.cmake)

# allow user to run tests with `make test` or `ctest`
catch_discover_tests(tests)
