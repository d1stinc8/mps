#include "circuits.hpp"
#include <gtest/gtest.h>

// Convention: U index = 2*s(q1) + s(q2). First argument is the first factor.
TEST(Exact, ArgumentOrder) {
  {
    ExactSim e(3);
    e.apply_1q(0, gates::X());
    e.apply_2q(0, 2, gates::CNOT());
    EXPECT_NEAR(std::abs(e.psi[0b101]), 1.0, 1e-12);
  } // control 0 on: target flips
  {
    ExactSim e(3);
    e.apply_1q(2, gates::X());
    e.apply_2q(2, 0, gates::CNOT());
    EXPECT_NEAR(std::abs(e.psi[0b101]), 1.0, 1e-12);
  } // control is qubit 2
  {
    ExactSim e(3);
    e.apply_1q(2, gates::X());
    e.apply_2q(0, 2, gates::CNOT());
    EXPECT_NEAR(std::abs(e.psi[0b001]), 1.0, 1e-12);
  } // control off: no flip
}

TEST(Exact, NonAdjacentGHZ) {
  ExactSim e(3);
  e.apply_1q(0, gates::H());
  e.apply_2q(0, 2, gates::CNOT()); // qubit 1 untouched: (|000> + |101>)/sqrt2
  const double h = 1.0 / std::sqrt(2.0);
  EXPECT_NEAR(e.psi[0b000].real(), h, 1e-12);
  EXPECT_NEAR(e.psi[0b101].real(), h, 1e-12);
  EXPECT_NEAR(e.norm2(), 1.0, 1e-12);
}

TEST(Exact, GHZEntropyIsOneBit) {
  const int N = 6;
  ExactSim e(N);
  e.apply_1q(0, gates::H());
  for (int i = 0; i + 1 < N; ++i)
    e.apply_2q(i, i + 1, gates::CNOT());
  for (int b = 0; b + 1 < N; ++b)
    EXPECT_NEAR(e.entropy(b), 1.0, 1e-10);
}

// Independent implementations (MPS vs dense) must agree on state and entropy.
TEST(Exact, MPSAgreesOnLayeredCircuit) {
  const int N = 8;
  MPS psi(N);
  auto ref = zero_state(N);
  layered_circuit(psi, ref, 5, 11);
  ExactSim e(N);
  e.psi = ref;
  EXPECT_NEAR(e.norm2(), 1.0, 1e-10);
  EXPECT_NEAR(e.fidelity(psi.to_statevector()), 1.0, 1e-10);
  auto S = psi.entropies();
  for (int b = 0; b + 1 < N; ++b)
    EXPECT_NEAR(S[b], e.entropy(b), 1e-8);
}
