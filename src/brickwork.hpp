#pragma once
#include "gates.hpp"
#include <random>

// One fixed random circuit for any backend. Same seed = same angles.
//   apply1(q, U): 1q gate.   apply2(q, U): 2q gate on (q, q+1).
template <class F1, class F2>
void brickwork(int N, int layers, unsigned seed, F1 apply1, F2 apply2) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> ang(0, 6.28318530717958647692);
  for (int L = 0; L < layers; ++L) {
    for (int q = 0; q < N; ++q) {
      apply1(q, gates::Rx(ang(rng)));
      apply1(q, gates::Rz(ang(rng)));
    }
    for (int q = L % 2; q + 1 < N; q += 2)
      apply2(q, gates::CZ());
  }
}
