import sys
import matplotlib.pyplot as plt
import numpy as np

d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
floor = 1e-16   # log axis cannot show exactly 0
plt.plot(d["chi"], np.maximum(1 - d["true_fidelity"], floor), "o-", label="true (vs exact)")
plt.plot(d["chi"], np.maximum(1 - d["est_fidelity"], floor), "s--", label="estimate prod(1-w)")
plt.xscale("log", base=2)
plt.yscale("log")
plt.xlabel("chi_max")
plt.ylabel("infidelity")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.savefig("convergence.png", dpi=150, bbox_inches="tight")
