#include "brickwork.hpp"
#include "exact.hpp"
#include "mps.hpp"
#include <gtest/gtest.h>

// 2^(N/2) is the largest possible bond: exact there, lossy below.
TEST(Convergence, ExactAtFullChi) {
  const int N = 8, layers = 6;
  ExactSim ex(N);
  brickwork(
      N, layers, 1, [&](int q, const Mat2 &U) { ex.apply_1q(q, U); },
      [&](int q, const Mat4 &U) { ex.apply_2q(q, q + 1, U); });
  double F[2];
  double est[2];
  const int chis[2] = {2, 16};
  for (int k = 0; k < 2; ++k) {
    MPS psi(N);
    brickwork(
        N, layers, 1, [&](int q, const Mat2 &U) { psi.apply_1q(q, U); },
        [&](int q, const Mat4 &U) { psi.apply_2q(q, U, chis[k]); });
    F[k] = ex.fidelity(psi.to_statevector());
    est[k] = psi.fidelity;
  }
  EXPECT_LT(F[0], 0.99);         // chi=2 really truncates
  EXPECT_NEAR(F[1], 1.0, 1e-10); // chi=16 is exact
  EXPECT_NEAR(est[1], 1.0, 1e-10);
}
