#include "circuits.hpp"
#include <gtest/gtest.h>

TEST(Canonical, MovingCenterKeepsState) {
  const int N = 6;
  MPS psi(N);
  auto ref = zero_state(N);
  random_circuit(psi, ref, 80, 3);
  for (int c : {5, 0, 3, 2}) {
    psi.move_center(c);
    auto v = psi.to_statevector();
    for (size_t i = 0; i < ref.size(); ++i)
      EXPECT_NEAR(std::abs(v[i] - ref[i]), 0.0, 1e-10);
  }
}

TEST(Canonical, IsometryConditions) {
  const int N = 6;
  MPS psi(N);
  auto ref = zero_state(N);
  random_circuit(psi, ref, 80, 4);
  psi.move_center(3);
  for (int i = 0; i < 3; ++i) {
    Eigen::MatrixXcd L = psi.sites[i].left_mat();
    EXPECT_NEAR(
        (L.adjoint() * L - Eigen::MatrixXcd::Identity(L.cols(), L.cols()))
            .norm(),
        0.0, 1e-10);
  }
  for (int i = 4; i < N; ++i) {
    Eigen::MatrixXcd R = psi.sites[i].right_mat();
    EXPECT_NEAR(
        (R * R.adjoint() - Eigen::MatrixXcd::Identity(R.rows(), R.rows()))
            .norm(),
        0.0, 1e-10);
  }
}

TEST(Canonical, Norm) {
  const int N = 6;
  MPS psi(N);
  auto ref = zero_state(N);
  random_circuit(psi, ref, 60, 5);
  psi.move_center(2);
  EXPECT_NEAR(psi.norm(), 1.0, 1e-10);
  for (cplx &a : psi.sites[2].data)
    a *= 2.0;
  EXPECT_NEAR(psi.norm(), 2.0, 1e-10);
  psi.normalize();
  EXPECT_NEAR(psi.norm(), 1.0, 1e-12);
}

// One truncation on a canonical state: fidelity estimate must equal true
// fidelity.
TEST(Canonical, FidelityExactForSingleTruncation) {
  const int N = 6;
  MPS psi(N);
  auto ref = zero_state(N);
  layered_circuit(psi, ref, 6, 6);
  ASSERT_GT(psi.bond_dims()[2], 2);
  psi.apply_2q(2, gates::CZ(), /*chi_max=*/2);
  ref = dense_2q(ref, N, 2, gates::CZ());
  auto got = psi.to_statevector();
  cplx ov = 0;
  double ng = 0, nr = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    ov += std::conj(ref[i]) * got[i];
    ng += std::norm(got[i]);
    nr += std::norm(ref[i]);
  }
  EXPECT_LT(psi.fidelity, 0.999); // truncation really happened
  EXPECT_NEAR(std::norm(ov) / (ng * nr), psi.fidelity, 1e-10);
}

TEST(Entropy, ProductIsZero) {
  MPS psi(4);
  for (double S : psi.entropies())
    EXPECT_NEAR(S, 0.0, 1e-12);
}

TEST(Entropy, GHZIsOneBitPerBond) {
  const int N = 5;
  MPS psi(N);
  psi.apply_1q(0, gates::H());
  for (int i = 0; i + 1 < N; ++i) { // CNOT(i,i+1) = H CZ H on target
    psi.apply_1q(i + 1, gates::H());
    psi.apply_2q(i, gates::CZ());
    psi.apply_1q(i + 1, gates::H());
  }
  for (double S : psi.entropies())
    EXPECT_NEAR(S, 1.0, 1e-10);
}
