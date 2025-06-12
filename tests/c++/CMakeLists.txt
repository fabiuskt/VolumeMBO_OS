# The Catch test executable
add_executable(tests priority_queue_t.cpp median_fitter_t.cpp)
target_link_libraries(tests PUBLIC volumembo Catch2::Catch2WithMain)
