import sys
import matplotlib.pyplot as plt
import numpy as np

d = np.genfromtxt(sys.argv[1], delimiter=",", names=True)
for chi in np.unique(d["chi"]):
    m = d["chi"] == chi
    label = "unlimited" if chi == 0 else f"chi={int(chi)}"
    (line,) = plt.plot(d["t"][m], d["S_mid"][m], label=label)
    if chi > 0:  # a capped MPS cannot exceed log2(chi) bits
        plt.axhline(np.log2(chi), color=line.get_color(), ls=":", alpha=0.5)
plt.xlabel("t  (1/Omega)")
plt.ylabel("mid-chain entropy (bits)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("quench_entropy.png", dpi=150, bbox_inches="tight")
