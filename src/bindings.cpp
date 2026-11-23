#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <optional>
#include <random>
#include <string>

#include "mps.hpp"
#include "rydberg.hpp"

namespace py = pybind11;
using CArr = py::array_t<cplx, py::array::c_style | py::array::forcecast>;

// MPS plus its own RNG: sample() and measure() are reproducible from one seed.
struct PyMPS : MPS {
  std::mt19937_64 rng;
  PyMPS(int n, std::optional<uint64_t> seed)
      : MPS(n), rng(seed ? *seed : std::random_device{}()) {}
};

// Release builds disable assert(), so every index is checked here instead.
static void check_site(const MPS &s, int q) {
  if (q < 0 || q >= s.N)
    throw py::index_error("qubit " + std::to_string(q) +
                          " out of range for N=" + std::to_string(s.N));
}

static Mat2 to_mat2(const CArr &a) {
  if (a.ndim() != 2 || a.shape(0) != 2 || a.shape(1) != 2)
    throw py::value_error("expected a 2x2 matrix");
  auto r = a.unchecked<2>();
  Mat2 U;
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      U[i][j] = r(i, j);
  return U;
}

static Mat4 to_mat4(const CArr &a) {
  if (a.ndim() != 2 || a.shape(0) != 4 || a.shape(1) != 4)
    throw py::value_error("expected a 4x4 matrix");
  auto r = a.unchecked<2>();
  Mat4 U;
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
      U[i][j] = r(i, j);
  return U;
}

// None -> unlimited bond dimension.
static int cap(std::optional<int> chi) {
  if (!chi)
    return 1 << 30;
  if (*chi < 1)
    throw py::value_error("chi_max must be >= 1");
  return *chi;
}

PYBIND11_MODULE(_mps, m) {
  m.doc() =
      "Matrix Product State simulator for neutral-atom circuits (C++ core)";

  py::class_<PyMPS>(m, "MPS")
      .def(py::init([](int n, std::optional<uint64_t> seed) {
             if (n < 1)
               throw py::value_error("n must be >= 1");
             return PyMPS(n, seed);
           }),
           py::arg("n"), py::arg("seed") = py::none())
      .def_property_readonly("n", [](const PyMPS &s) { return s.N; })
      // Product of (1 - discarded weight) over all truncations. Exact for
      // canonical form.
      .def_property_readonly("fidelity",
                             [](const PyMPS &s) { return s.fidelity; })
      .def_property_readonly("discarded_weights",
                             [](const PyMPS &s) { return s.discarded_weights; })
      .def("bond_dims", [](const PyMPS &s) { return s.bond_dims(); })
      .def(
          "apply_1q",
          [](PyMPS &s, int q, const CArr &U) {
            check_site(s, q);
            s.apply_1q(q, to_mat2(U));
          },
          py::arg("q"), py::arg("U"))
      // Any pair q1 != q2. U index = 2*s(q1) + s(q2). Non-adjacent pairs use
      // SWAPs.
      .def(
          "apply_2q",
          [](PyMPS &s, int q1, int q2, const CArr &U,
             std::optional<int> chi_max, double cutoff) {
            check_site(s, q1);
            check_site(s, q2);
            if (q1 == q2)
              throw py::value_error("q1 and q2 must differ");
            s.apply_2q_any(q1, q2, to_mat4(U), cap(chi_max), cutoff);
          },
          py::arg("q1"), py::arg("q2"), py::arg("U"),
          py::arg("chi_max") = py::none(), py::arg("cutoff") = 1e-14)
      .def("to_statevector",
           [](const PyMPS &s) {
             if (s.N > 25)
               throw py::value_error(
                   "to_statevector needs N <= 25 (2^N amplitudes)");
             auto v = s.to_statevector();
             return py::array_t<cplx>(v.size(), v.data()); // copies
           })
      .def("norm", [](const PyMPS &s) { return s.norm(); })
      .def("normalize", [](PyMPS &s) { s.normalize(); })
      .def(
          "prob1",
          [](PyMPS &s, int q) {
            check_site(s, q);
            return s.prob1(q);
          },
          py::arg("q"))
      .def("densities",
           [](PyMPS &s) {
             auto n = s.densities();
             return py::array_t<double>(n.size(), n.data());
           })
      .def(
          "entropy",
          [](PyMPS &s, int bond) {
            if (bond < 0 || bond + 1 >= s.N)
              throw py::index_error("bond out of range");
            return s.entropy(bond);
          },
          py::arg("bond"))
      .def("entropies", [](PyMPS &s) { return s.entropies(); })
      // (shots, N) uint8 array. Qubit q = column q. Does not collapse the
      // state.
      .def(
          "sample",
          [](PyMPS &s, int shots) {
            if (shots < 1)
              throw py::value_error("shots must be >= 1");
            py::array_t<uint8_t> out({(py::ssize_t)shots, (py::ssize_t)s.N});
            auto r = out.mutable_unchecked<2>();
            for (int k = 0; k < shots; ++k) {
              auto bits = s.sample(s.rng);
              for (int q = 0; q < s.N; ++q)
                r(k, q) = (uint8_t)bits[q];
            }
            return out;
          },
          py::arg("shots") = 1)
      // Projective measurement. Collapses the state. Returns 0 or 1.
      .def(
          "measure",
          [](PyMPS &s, int q) {
            check_site(s, q);
            return s.measure(q, s.rng);
          },
          py::arg("q"));

  // Evolve under the Rydberg chain H(t) with omega(site, t) and delta(site, t)
  // callables.
  m.def(
      "evolve_rydberg",
      [](PyMPS &psi, double V_nn, int R,
         std::function<double(int, double)> omega,
         std::function<double(int, double)> delta, double t0, double T,
         int steps, std::optional<int> chi_max, double cutoff) {
        if (R < 1)
          throw py::value_error("R must be >= 1");
        if (steps < 1)
          throw py::value_error("steps must be >= 1");
        auto H =
            make_vdw_chain(psi.N, V_nn, R, std::move(omega), std::move(delta));
        evolve(psi, H, t0, T, steps, cap(chi_max), cutoff);
      },
      py::arg("psi"), py::arg("V_nn"), py::arg("R"), py::arg("omega"),
      py::arg("delta"), py::arg("t0"), py::arg("T"), py::arg("steps"),
      py::arg("chi_max") = py::none(), py::arg("cutoff") = 1e-14);

  m.def(
      "z2_order", [](const std::vector<double> &n) { return z2_order(n); },
      py::arg("densities"));
}
