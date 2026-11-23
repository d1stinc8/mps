// Adiabatic Z2 preparation in a Rydberg chain. Sweep Delta from -D0 to +D0.
// usage: ./z2_sweep [N=11] [R=1] [dt=0.02] [chi=64]
// CSV: T,order,est_fidelity,max_bond,seconds
#include "mps.hpp"
#include "rydberg.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

static const double PI = 3.14159265358979323846;

int main(int argc, char **argv) {
  const int N =
      argc > 1 ? std::atoi(argv[1]) : 11; // odd: unique Z2 ground state
  const int R = argc > 2 ? std::atoi(argv[2]) : 1;
  const double dt = argc > 3 ? std::atof(argv[3]) : 0.02;
  const int chi = argc > 4 ? std::atoi(argv[4]) : 64;
  const double Om = 1.0, D0 = 3.0, Vnn = 10.0;

  std::cout << "T,order,est_fidelity,max_bond,seconds\n";
  for (double T : {16.0, 32.0, 64.0, 96.0, 128.0, 192.0}) {
    // sin^2 ramp over the first and last 10% of T, flat in between.
    auto Omega = [=](int, double t) {
      const double tr = 0.1 * T;
      const double s = (t < tr)       ? std::sin(0.5 * PI * t / tr)
                       : (t > T - tr) ? std::sin(0.5 * PI * (T - t) / tr)
                                      : 1.0;
      return Om * s * s;
    };
    auto Delta = [=](int, double t) { return -D0 + 2.0 * D0 * t / T; };
    auto H = make_vdw_chain(N, Vnn, R, Omega, Delta);

    MPS psi(N);
    const auto t0 = std::chrono::steady_clock::now();
    evolve(psi, H, 0.0, T, int(std::round(T / dt)), chi);
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    const auto d = psi.bond_dims();
    std::cout << T << "," << z2_order(psi.densities()) << "," << psi.fidelity
              << "," << *std::max_element(d.begin(), d.end()) << "," << sec
              << "\n";
  }
}
