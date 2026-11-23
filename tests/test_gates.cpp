#include "dense.hpp"
#include "mps.hpp"
#include <gtest/gtest.h>

static bool unitary(const Mat2 &U) { // U^dagger U == I
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) {
      cplx s = 0;
      for (int k = 0; k < 2; ++k)
        s += std::conj(U[k][i]) * U[k][j];
      if (std::abs(s - (i == j ? 1.0 : 0.0)) > 1e-12)
        return false;
    }
  return true;
}

TEST(Gates, Unitary) {
  EXPECT_TRUE(unitary(gates::H()));
  EXPECT_TRUE(unitary(gates::X()));
  EXPECT_TRUE(unitary(gates::Z()));
  EXPECT_TRUE(unitary(gates::Rx(0.37)));
  EXPECT_TRUE(unitary(gates::Rz(1.91)));
}

TEST(Gates, MatchesDense) {
  const int N = 4;
  struct Op {
    int q;
    Mat2 U;
  };
  std::vector<Op> circuit = {
      {0, gates::H()},      {1, gates::Rx(0.7)}, {2, gates::X()},
      {3, gates::Rz(1.3)},  {1, gates::H()},     {0, gates::Rx(2.1)},
      {2, gates::Rz(-0.4)}, {3, gates::Rx(0.9)},
  };
  MPS psi(N);
  std::vector<cplx> ref(1 << N, 0.0);
  ref[0] = 1.0;
  for (auto &op : circuit) {
    psi.apply_1q(op.q, op.U);
    ref = dense_1q(ref, N, op.q, op.U);
  }
  auto got = psi.to_statevector();
  for (size_t i = 0; i < ref.size(); ++i)
    EXPECT_NEAR(std::abs(got[i] - ref[i]), 0.0, 1e-12);
}
