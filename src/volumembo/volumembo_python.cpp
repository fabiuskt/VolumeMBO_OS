#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "volumembo/median_fitter.hpp"
#include "volumembo/pybind11_numpy_interop.hpp"
#include "volumembo/span2d.hpp"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

namespace volumembo {

PYBIND11_MODULE(_volumembo, m)
{
  m.doc() = "Python Bindings for volumembo";

  m.def(
    "fit_median_cpp",
    [](py::array_t<double, py::array::c_style | py::array::forcecast> u_np,
       std::vector<unsigned int> lower_limit,
       std::vector<unsigned int> upper_limit) {
      if (u_np.ndim() != 2)
        throw std::invalid_argument("u must be a 2D array");

      std::size_t N = u_np.shape(0);
      std::size_t M = u_np.shape(1);

      if (lower_limit.size() != M || upper_limit.size() != M)
        throw std::invalid_argument(
          "Volume limits must match number of clusters");

      // Wrap in custom Span2D type
      Span2D<const double> u_view(u_np.data(), N, M);

      // Run the fitter
      VolumeMedianFitter fitter(u_view, lower_limit, upper_limit);
      return fitter.fit();
    },
    py::arg("u"),
    py::arg("lower_limit"),
    py::arg("upper_limit"),
    "Fit median using the C++ priority queue implementation (zero-copy)");
}

} // namespace volumembo
