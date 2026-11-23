#include "circuits.hpp"
#include <gtest/gtest.h>

TEST(Sampling, MatchesBornDistribution) {
  const int N = 6;
  MPS psi(N);
  auto ref = zero_state(N);
  layered_circuit(psi, ref, 4, 31);
  std::mt19937_64 rng(123);
  const int shots = 100000;
  std::vector<double> hist(size_t(1) << N, 0.0);
  for (int k = 0; k < shots; ++k) {
    auto s = psi.sample(rng);
    size_t idx = 0;
    for (int q = 0; q < N; ++q)
      idx = (idx << 1) | size_t(s[q]);
    hist[idx] += 1.0;
  }
  for (size_t i = 0; i < hist.size(); ++i)
    EXPECT_NEAR(hist[i] / shots, std::norm(ref[i]), 0.005) << "outcome " << i;

  ExactSim e(N); // sampling only changed the gauge
  e.psi = ref;
  EXPECT_NEAR(e.fidelity(psi.to_statevector()), 1.0, 1e-10);
}

TEST(Sampling, GHZOnlyAllEqual) {
  const int N = 8;
  MPS psi(N);
  psi.apply_1q(0, gates::H());
  for (int i = 0; i + 1 < N; ++i)
    psi.apply_2q(i, gates::CNOT());
  std::mt19937_64 rng(7);
  int ones = 0;
  for (int k = 0; k < 1000; ++k) {
    auto s = psi.sample(rng);
    for (int q = 1; q < N; ++q)
      ASSERT_EQ(s[q], s[0]);
    ones += s[0];
  }
  EXPECT_GT(ones, 400);
  EXPECT_LT(ones, 600);
}

TEST(Measure, Prob1MatchesExact) {
  const int N = 6;
  MPS psi(N);
  auto ref = zero_state(N);
  layered_circuit(psi, ref, 4, 41);
  ExactSim e(N);
  e.psi = ref;
  for (int q = 0; q < N; ++q) {
    double want = 0;
    for (size_t i = 0; i < ref.size(); ++i)
      if (i & e.mask(q))
        want += std::norm(ref[i]);
    EXPECT_NEAR(psi.prob1(q), want, 1e-10) << "q=" << q;
  }
  EXPECT_NEAR(e.fidelity(psi.to_statevector()), 1.0, 1e-10);
}

// Measure one GHZ qubit: every other qubit is now fixed to the same value.
TEST(Measure, GHZCollapse) {
  int ones = 0;
  for (unsigned seed = 0; seed < 20; ++seed) {
    MPS psi(5);
    psi.apply_1q(0, gates::H());
    for (int i = 0; i < 4; ++i)
      psi.apply_2q(i, gates::CNOT());
    std::mt19937_64 rng(seed);
    const int m = psi.measure(2, rng);
    ones += m;
    for (int q = 0; q < 5; ++q)
      EXPECT_NEAR(psi.prob1(q), double(m), 1e-12);
    EXPECT_NEAR(psi.norm(), 1.0, 1e-12);
  }
  EXPECT_GT(ones, 0); // both outcomes occur
  EXPECT_LT(ones, 20);
}
