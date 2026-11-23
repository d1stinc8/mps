// Entanglement growth after a quench from |0...0>, constant H.
// usage: ./quench_entropy [N=12] [R=1] [tmax=8] [dt=0.02]
// CSV: chi,t,S_mid,max_bond,est_fidelity     (chi = 0 means unlimited)
#include "mps.hpp"
#include "rydberg.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>

int main(int argc, char **argv) {
  const int N = argc > 1 ? std::atoi(argv[1]) : 12;
  const int R = argc > 2 ? std::atoi(argv[2]) : 1;
  const double tmax = argc > 3 ? std::atof(argv[3]) : 8.0;
  const double dt = argc > 4 ? std::atof(argv[4]) : 0.02;
  const int rec = 5; // Strang steps between records: record every rec*dt

  auto H = make_vdw_chain(
      N, 10.0, R, [](int, double) { return 1.0; },
      [](int, double) { return 0.0; });

  std::cout << "chi,t,S_mid,max_bond,est_fidelity\n";
  for (int chi : {2, 4, 8, 0}) {
    const int cap = chi > 0 ? chi : (1 << 30);
    MPS psi(N);
    for (int k = 0; k * rec * dt <= tmax + 1e-9; ++k) {
      if (k > 0)
        evolve(psi, H, 0.0, rec * dt, rec, cap); // H is static: t0 irrelevant
      const auto d = psi.bond_dims();
      std::cout << chi << "," << k * rec * dt << "," << psi.entropy(N / 2 - 1)
                << "," << *std::max_element(d.begin(), d.end()) << ","
                << psi.fidelity << "\n";
    }
  }
}
