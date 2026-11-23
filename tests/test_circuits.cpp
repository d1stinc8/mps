#include "circuits.hpp"
#include <gtest/gtest.h>

static const double PI = 3.14159265358979323846;

// Standard QFT (Nielsen & Chuang). Qubit 0 = most significant bit.
// Controlled phases span long distances, so this also exercises the swap
// network.
static void qft(MPS &psi) {
  const int n = psi.N;
  for (int j = 0; j < n; ++j) {
    psi.apply_1q(j, gates::H());
    for (int k = j + 1; k < n; ++k)
      psi.apply_2q_any(k, j, gates::CPhase(PI / double(1 << (k - j))));
  }
  for (int i = 0; i < n / 2; ++i) // bit reversal
    psi.apply_2q_any(i, n - 1 - i, gates::SWAP());
}

// QFT|x> = 2^{-n/2} sum_y exp(2 pi i x y / 2^n) |y>. A product state.
TEST(QFT, MatchesAnalyticFormula) {
  const int n = 5;
  const size_t D = size_t(1) << n;
  for (size_t x : {0u, 5u, 13u, 31u}) {
    MPS psi(n);
    for (int q = 0; q < n; ++q)
      if ((x >> (n - 1 - q)) & 1)
        psi.apply_1q(q, gates::X());
    qft(psi);
    auto v = psi.to_statevector();
    for (size_t y = 0; y < D; ++y) {
      cplx want = std::polar(1.0 / std::sqrt(double(D)),
                             2 * PI * double(x * y) / double(D));
      EXPECT_NEAR(std::abs(v[y] - want), 0.0, 1e-10) << "x=" << x << " y=" << y;
    }
    for (double S : psi.entropies())
      EXPECT_NEAR(S, 0.0, 1e-8); // product state
  }
}

TEST(GHZ, AmplitudesAndBonds) {
  const int N = 8;
  MPS psi(N);
  psi.apply_1q(0, gates::H());
  for (int i = 0; i + 1 < N; ++i)
    psi.apply_2q(i, gates::CNOT());
  auto v = psi.to_statevector();
  const double h = 1.0 / std::sqrt(2.0);
  EXPECT_NEAR(v.front().real(), h, 1e-12);
  EXPECT_NEAR(v.back().real(), h, 1e-12);
  double rest = 0;
  for (size_t i = 1; i + 1 < v.size(); ++i)
    rest += std::norm(v[i]);
  EXPECT_NEAR(rest, 0.0, 1e-20);
  for (int d : psi.bond_dims())
    EXPECT_EQ(d, 2);
}

// Truncation shrinks theta; apply_2q rescales to compensate.
// With canonical form the whole state's norm must stay 1.
TEST(Truncation, NormSurvives) {
  const int N = 6;
  MPS psi(N);
  for (int q = 0; q < N; ++q)
    psi.apply_1q(q, gates::H());
  for (int q = 0; q + 1 < N; ++q)
    psi.apply_2q(q, gates::CZ(), /*chi_max=*/1);
  EXPECT_LT(psi.fidelity, 1.0); // truncation really happened
  EXPECT_NEAR(psi.norm(), 1.0, 1e-10);
  ExactSim e(N);
  e.psi = psi.to_statevector();
  EXPECT_NEAR(e.norm2(), 1.0, 1e-10);
}
