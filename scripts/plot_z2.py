import sys
import matplotlib.pyplot as plt
import numpy as np

d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
plt.plot(d["T"], d["order"], "o-")
plt.xlabel("sweep time T  (1/Omega)")
plt.ylabel("Z2 order parameter")
plt.ylim(-0.1, 1.05)
plt.grid(True, alpha=0.3)
plt.savefig("z2.png", dpi=150, bbox_inches="tight")
