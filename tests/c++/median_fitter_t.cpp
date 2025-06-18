#include "catch2/catch_test_macros.hpp"
#include "volumembo/priority_queue.hpp"
#include <catch2/catch_approx.hpp>

#include "volumembo/median_fitter.hpp"
#include "volumembo/span2d.hpp"

#include <numeric>
#include <vector>

TEST_CASE("Fit median 2D")
{
  constexpr unsigned int N = 6; // number of points
  constexpr unsigned int M = 2; // number of clusters

  // Lower and upper limits
  const std::vector<unsigned int> lower_limit = { 3, 3 };
  const std::vector<unsigned int> upper_limit = { 3, 3 };

  SECTION("2D Grow cluster 0")
  {
    // Example u matrix: 6 points in 2D space (flattened row-major)
    const std::vector<double> u = { 0.28, 0.72, 0.25, 0.75, 0.2,  0.8,
                                    0.15, 0.85, 0.1,  0.9,  0.05, 0.95 };

    volumembo::Span2D<const double> u_span(u, N, M);
    volumembo::VolumeMedianFitter fitter(u_span, lower_limit, upper_limit);

    std::vector<double> result = fitter.fit();

    // Check that each entry is in [0, 1]
    for (double val : result) {
      REQUIRE(val >= 0.0);
      REQUIRE(val <= 1.0);
    }

    // Check that the sum is approximately 1.0
    double sum = std::accumulate(result.begin(), result.end(), 0.0);
    REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-8));
  }

  SECTION("2D Shrink cluster 0")
  {
    // Example u matrix: 6 points in 2D space (flattened row-major)

    const std::vector<double> u = { 0.72, 0.28, 0.75, 0.25, 0.8,  0.2,
                                    0.85, 0.15, 0.9,  0.1,  0.95, 0.05 };

    volumembo::Span2D<const double> u_span(u, N, M);
    volumembo::VolumeMedianFitter fitter(u_span, lower_limit, upper_limit);

    std::vector<double> result = fitter.fit();

    // Check that each entry is in [0, 1]
    for (double val : result) {
      REQUIRE(val >= 0.0);
      REQUIRE(val <= 1.0);
    }

    // Check that the sum is approximately 1.0
    double sum = std::accumulate(result.begin(), result.end(), 0.0);
    REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-8));
  }
}

TEST_CASE("Fit median 3D")
{
  constexpr unsigned int N = 9; // number of points
  constexpr unsigned int M = 3; // number of clusters

  // Lower and upper limits
  const std::vector<unsigned int> lower_limit = { 3, 3, 3 };
  const std::vector<unsigned int> upper_limit = { 3, 3, 3 };

  SECTION("3D Grow cluster 0")
  {
    // Example u matrix: 6 points in 3D space (flattened row-major)
    const std::vector<double> u = {
      0.8, 0.1, 0.1, 0.7, 0.2, 0.1, 0.6, 0.3, 0.1, 0.1, 0.7, 0.2, 0.2, 0.6,
      0.2, 0.3, 0.1, 0.6, 0.4, 0.1, 0.5, 0.1, 0.2, 0.7, 0.1, 0.1, 0.8
    };

    volumembo::Span2D<const double> u_span(u, N, M);
    volumembo::VolumeMedianFitter fitter(u_span, lower_limit, upper_limit);

    std::vector<double> result = fitter.fit();

    // Check that each entry is in [0, 1]
    for (double val : result) {
      REQUIRE(val >= 0.0);
      REQUIRE(val <= 1.0);
    }

    // Check that the sum is approximately 1.0
    double sum = std::accumulate(result.begin(), result.end(), 0.0);
    REQUIRE(sum == Catch::Approx(1.0).epsilon(1e-8));
  }
}
