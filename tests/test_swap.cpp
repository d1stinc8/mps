#include "circuits.hpp"
#include <gtest/gtest.h>

// Every ordered pair (q1,q2), asymmetric gate, entangled start state.
// If the argument order or the swap chain is wrong, CNOT exposes it.
TEST(SwapNetwork, AllPairsMatchExact) {
  const int N = 6;
  MPS base(N);
  auto ref = zero_state(N);
  layered_circuit(base, ref, 4, 21);
  for (int q1 = 0; q1 < N; ++q1)
    for (int q2 = 0; q2 < N; ++q2) {
      if (q1 == q2)
        continue;
      MPS p = base;
      ExactSim e(N);
      e.psi = ref;
      p.apply_2q_any(q1, q2, gates::CNOT());
      e.apply_2q(q1, q2, gates::CNOT());
      EXPECT_NEAR(e.fidelity(p.to_statevector()), 1.0, 1e-10)
          << "q1=" << q1 << " q2=" << q2;
    }
}

TEST(SwapNetwork, RandomLongRangeCircuit) {
  const int N = 7;
  std::mt19937 rng(5);
  std::uniform_real_distribution<double> ang(0, 6.28);
  std::uniform_int_distribution<int> site(0, N - 1), kind(0, 2);
  MPS psi(N);
  ExactSim e(N);
  for (int s = 0; s < 60; ++s) {
    int a = site(rng), b = site(rng);
    switch (kind(rng)) {
    case 0: {
      auto U = gates::Rx(ang(rng));
      psi.apply_1q(a, U);
      e.apply_1q(a, U);
      break;
    }
    case 1: {
      auto U = gates::Rz(ang(rng));
      psi.apply_1q(a, U);
      e.apply_1q(a, U);
      break;
    }
    default:
      if (a == b)
        break;
      psi.apply_2q_any(a, b, gates::CNOT());
      e.apply_2q(a, b, gates::CNOT());
    }
  }
  EXPECT_NEAR(e.fidelity(psi.to_statevector()), 1.0, 1e-10);
  EXPECT_NEAR(psi.fidelity, 1.0, 1e-10);
}

// Every bond is only ever cut by apply_2q, so chi_max=2 caps all of them.
TEST(SwapNetwork, ChiCapHoldsAcrossSwaps) {
  const int N = 6;
  MPS psi(N);
  for (int q = 0; q < N; ++q)
    psi.apply_1q(q, gates::H());
  psi.apply_2q_any(0, 5, gates::CZ(), /*chi_max=*/2);
  psi.apply_2q_any(4, 1, gates::CNOT(), 2);
  for (int d : psi.bond_dims())
    EXPECT_LE(d, 2);
  EXPECT_LE(psi.fidelity, 1.0);
}
