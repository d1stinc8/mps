#pragma once
#include "gates.hpp"
#include "mps.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <vector>

// H(t) = sum_i (Omega_i(t)/2) X_i - sum_i Delta_i(t) n_i + sum_{i<j} V_ij n_i
// n_j n = |1><1| (Rydberg state). Energies in units of Omega_max, time in
// 1/Omega_max.
struct RydbergChain {
  int N;
  std::vector<double> V; // V[i]: sites (i, i+1)
  std::vector<std::vector<double>>
      Vlong; // Vlong[d-2][i]: sites (i, i+d), d = 2..R
  std::function<double(int, double)> Omega, Delta; // (site, time) -> rate
};

// Van der Waals chain: V_ij = C6 / r^6 with r = a*d  =>  V_ij = V_nn / d^6.
// Couplings with d > R are dropped (the cutoff).
inline RydbergChain make_vdw_chain(int N, double V_nn, int R,
                                   std::function<double(int, double)> Omega,
                                   std::function<double(int, double)> Delta) {
  RydbergChain H;
  H.N = N;
  H.Omega = std::move(Omega);
  H.Delta = std::move(Delta);
  H.V.assign(N - 1, V_nn);
  for (int d = 2; d <= R; ++d)
    H.Vlong.push_back(
        std::vector<double>(std::max(0, N - d), V_nn / std::pow(double(d), 6)));
  return H;
}

// Z2 order parameter from Rydberg densities n_i: (1/N) sum (-1)^i (2 n_i - 1).
// = +1 for |1010...1> (odd N: unique ground state, both ends excited).
inline double z2_order(const std::vector<double> &n) {
  double m = 0;
  for (size_t i = 0; i < n.size(); ++i)
    m += ((i % 2 == 0) ? 1.0 : -1.0) * (2.0 * n[i] - 1.0);
  return m / double(n.size());
}

// exp(-i dt (omega/2 X - delta n)) in closed form. n = (I - Z)/2, so
// h = -delta/2 I + (omega X + delta Z)/2.
inline Mat2 drive_gate(double omega, double delta, double dt) {
  const double w = std::hypot(omega, delta);
  const double th = 0.5 * w * dt;
  const double c = std::cos(th);
  const double s = (w > 0) ? std::sin(th) / w : 0.0; // sin(th)/w, safe at w = 0
  const cplx i(0, 1);
  const cplx ph = std::polar(1.0, 0.5 * delta * dt); // e^{+i delta dt / 2}
  Mat2 U;
  U[0][0] = ph * (c - i * s * delta);
  U[0][1] = ph * (-i * s * omega);
  U[1][0] = U[0][1];
  U[1][1] = ph * (c + i * s * delta);
  return U;
}

// One Strang step t -> t+dt. Backend-agnostic:
//   apply1(q, U): 1q gate on q.   apply2(q1, q2, U): 2q gate on any pair q1 <
//   q2.
// H1 (drive) and H2 (all interactions) are each exact products of gates.
template <class F1, class F2>
void strang_step_lr(const RydbergChain &H, double t, double dt, F1 apply1,
                    F2 apply2) {
  const double tm =
      t + 0.5 * dt; // midpoint: second order for time-dependent drive
  auto half_drive = [&] {
    for (int i = 0; i < H.N; ++i)
      apply1(i, drive_gate(H.Omega(i, tm), H.Delta(i, tm), 0.5 * dt));
  };
  half_drive();
  for (int i = 0; i + 1 < H.N; ++i)
    apply2(i, i + 1, gates::CPhase(-H.V[i] * dt));
  for (size_t k = 0; k < H.Vlong.size(); ++k) {
    const int d = int(k) + 2;
    for (int i = 0; i + d < H.N; ++i)
      apply2(i, i + d, gates::CPhase(-H.Vlong[k][i] * dt));
  }
  half_drive();
}

// Nearest-neighbor-only wrapper: apply2(q, U) on (q, q+1). Same signature as
// before.
template <class F1, class F2>
void strang_step(const RydbergChain &H, double t, double dt, F1 apply1,
                 F2 apply2) {
  assert(H.Vlong.empty());
  strang_step_lr(H, t, dt, apply1,
                 [&](int q, int, const Mat4 &U) { apply2(q, U); });
}

// Evolve an MPS from t0 to t0+T in `steps` Strang steps.
// chi_max / cutoff apply at every two-site gate, swaps included.
inline void evolve(MPS &psi, const RydbergChain &H, double t0, double T,
                   int steps, int chi_max = 1 << 30, double cutoff = 1e-14) {
  const double dt = T / steps;
  for (int k = 0; k < steps; ++k)
    strang_step_lr(
        H, t0 + k * dt, dt, [&](int q, const Mat2 &U) { psi.apply_1q(q, U); },
        [&](int a, int b, const Mat4 &U) {
          psi.apply_2q_any(a, b, U, chi_max, cutoff);
        });
}
