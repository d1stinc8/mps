#include "mps.hpp"
#include <iostream>

int main() {
  MPS psi(3);
  psi.apply_1q(0, gates::H()); // expect (|000> + |100>)/sqrt2

  auto v = psi.to_statevector();
  for (size_t i = 0; i < v.size(); ++i)
    std::cout << i << ": " << v[i] << "\n";
}
