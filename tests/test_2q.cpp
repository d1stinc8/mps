#include "dense.hpp"
#include "mps.hpp"
#include <gtest/gtest.h>
#include <random>

TEST(TwoQubit, BellState) {
  MPS psi(2);
  psi.apply_1q(0, gates::H());
  psi.apply_1q(1, gates::H());
  psi.apply_2q(0, gates::CZ());
  psi.apply_1q(1, gates::H()); // now (|00> + |11>)/sqrt2
  auto v = psi.to_statevector();
  const double h = 1.0 / std::sqrt(2.0);
  EXPECT_NEAR(v[0].real(), h, 1e-12);
  EXPECT_NEAR(v[3].real(), h, 1e-12);
  EXPECT_NEAR(std::abs(v[1]), 0.0, 1e-12);
  EXPECT_EQ(psi.bond_dims()[0], 2);
  EXPECT_NEAR(psi.fidelity, 1.0, 1e-12);
}

TEST(TwoQubit, TruncateBellToChi1) {
  // Both Schmidt values are 1/sqrt2, so dropping one loses weight 0.5.
  MPS psi(2);
  psi.apply_1q(0, gates::H());
  psi.apply_1q(1, gates::H());
  psi.apply_2q(0, gates::CZ(), /*chi_max=*/1);
  EXPECT_EQ(psi.bond_dims()[0], 1);
  EXPECT_NEAR(psi.discarded_weights.back(), 0.5, 1e-12);
  EXPECT_NEAR(psi.fidelity, 0.5, 1e-12);
}

TEST(TwoQubit, RandomCircuitMatchesDense) {
  const int N = 6;
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> ang(0, 6.28);
  std::uniform_int_distribution<int> site(0, N - 2), kind(0, 3);

  MPS psi(N);
  std::vector<cplx> ref(1 << N, 0.0);
  ref[0] = 1.0;
  for (int step = 0; step < 60; ++step) {
    int q = site(rng);
    switch (kind(rng)) {
    case 0: {
      auto U = gates::Rx(ang(rng));
      psi.apply_1q(q, U);
      ref = [&] { // inline dense 1q via 2q identity-free path
        const size_t mask = size_t(1) << (N - 1 - q);
        auto v = ref;
        for (size_t i = 0; i < v.size(); ++i)
          if (!(i & mask)) {
            cplx a0 = ref[i], a1 = ref[i | mask];
            v[i] = U[0][0] * a0 + U[0][1] * a1;
            v[i | mask] = U[1][0] * a0 + U[1][1] * a1;
          }
        return v;
      }();
      break;
    }
    case 1: {
      auto U = gates::Rz(ang(rng));
      psi.apply_1q(q, U);
      ref = [&] {
        const size_t mask = size_t(1) << (N - 1 - q);
        auto v = ref;
        for (size_t i = 0; i < v.size(); ++i)
          if (!(i & mask)) {
            cplx a0 = ref[i], a1 = ref[i | mask];
            v[i] = U[0][0] * a0 + U[0][1] * a1;
            v[i | mask] = U[1][0] * a0 + U[1][1] * a1;
          }
        return v;
      }();
      break;
    }
    default:
      psi.apply_2q(q, gates::CZ());
      ref = dense_2q(ref, N, q, gates::CZ());
      break;
    }
  }
  auto got = psi.to_statevector();
  for (size_t i = 0; i < ref.size(); ++i)
    EXPECT_NEAR(std::abs(got[i] - ref[i]), 0.0, 1e-10);
  EXPECT_NEAR(psi.fidelity, 1.0, 1e-10);
}
