#pragma once
#include "dense.hpp"
#include <random>

inline std::vector<cplx> zero_state(int N) {
  std::vector<cplx> v(size_t(1) << N, 0.0);
  v[0] = 1.0;
  return v;
}

// Random Rx/Rz/CZ circuit applied to both the MPS and a dense vector.
inline void random_circuit(MPS &psi, std::vector<cplx> &ref, int steps,
                           unsigned seed) {
  const int N = psi.N;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ang(0, 6.28);
  std::uniform_int_distribution<int> site(0, N - 1), kind(0, 2);
  for (int s = 0; s < steps; ++s) {
    int q = site(rng);
    switch (kind(rng)) {
    case 0: {
      auto U = gates::Rx(ang(rng));
      psi.apply_1q(q, U);
      ref = dense_1q(ref, N, q, U);
      break;
    }
    case 1: {
      auto U = gates::Rz(ang(rng));
      psi.apply_1q(q, U);
      ref = dense_1q(ref, N, q, U);
      break;
    }
    default:
      q = std::min(q, N - 2);
      psi.apply_2q(q, gates::CZ());
      ref = dense_2q(ref, N, q, gates::CZ());
    }
  }
}

// Brickwork: random 1q rotations on all sites, then CZ on alternating pairs.
inline void layered_circuit(MPS &psi, std::vector<cplx> &ref, int layers,
                            unsigned seed) {
  const int N = psi.N;
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ang(0, 6.28);
  for (int L = 0; L < layers; ++L) {
    for (int q = 0; q < N; ++q) {
      auto A = gates::Rx(ang(rng)), B = gates::Rz(ang(rng));
      psi.apply_1q(q, A);
      ref = dense_1q(ref, N, q, A);
      psi.apply_1q(q, B);
      ref = dense_1q(ref, N, q, B);
    }
    for (int q = L % 2; q + 1 < N; q += 2) {
      psi.apply_2q(q, gates::CZ());
      ref = dense_2q(ref, N, q, gates::CZ());
    }
  }
}
