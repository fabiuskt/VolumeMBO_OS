#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "volumembo/median_fitter.hpp"
#include "volumembo/pybind11_numpy_interop.hpp"

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

      size_t N = u_np.shape(0);
      size_t M = u_np.shape(1);
      if (lower_limit.size() != M || upper_limit.size() != M)
        throw std::invalid_argument(
          "Volume limits must match number of clusters");

      // Convert NumPy array to std::vector<std::vector<double>>
      const double* data = u_np.data();
      std::vector<std::vector<double>> u_vec(N, std::vector<double>(M));
      for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < M; ++j)
          u_vec[i][j] = data[i * M + j];

      volumembo::VolumeMedianFitter fitter(u_vec, lower_limit, upper_limit);
      return fitter.fit(); // returns std::vector<double>
    },
    py::arg("u"),
    py::arg("lower_limit"),
    py::arg("upper_limit"),
    "Fit median using the C++ priority queue implementation");
}

} // namespace volumembo
