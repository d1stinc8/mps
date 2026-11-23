// Fidelity vs chi on a random brickwork circuit. Prints CSV.
// usage: ./convergence [N=14] [layers=8]
#include "brickwork.hpp"
#include "exact.hpp"
#include "mps.hpp"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 14;
  const int layers = argc > 2 ? std::atoi(argv[2]) : 8;

  ExactSim ex(N);
  brickwork(
      N, layers, 1, [&](int q, const Mat2 &U) { ex.apply_1q(q, U); },
      [&](int q, const Mat4 &U) { ex.apply_2q(q, q + 1, U); });

  std::cout << "chi,true_fidelity,est_fidelity,max_bond,seconds\n";
  // Largest possible bond is 2^(N/2): chi at that value is exact.
  for (int chi = 1; chi <= (1 << (N / 2)); chi *= 2) {
    MPS psi(N);
    const auto t0 = std::chrono::steady_clock::now();
    brickwork(
        N, layers, 1, [&](int q, const Mat2 &U) { psi.apply_1q(q, U); },
        [&](int q, const Mat4 &U) { psi.apply_2q(q, U, chi); });
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    const auto d = psi.bond_dims();
    std::cout << chi << "," << ex.fidelity(psi.to_statevector()) << ","
              << psi.fidelity << "," << *std::max_element(d.begin(), d.end())
              << "," << sec << "\n";
  }
}
