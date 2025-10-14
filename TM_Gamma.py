"""
Recreate the figure 'LLE_Distribution.pdf' using Γ-based sensitivity measure
(Γ = average rate of ensemble separation) for the Tent Map with a = 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from Gamma import *
from Maps_1D import *
from scipy.stats import linregress


# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------

a = 1.999                  # control parameter
Ns = 1_000_000             # number of samples
xmin, xmax = 0.0, 1.0    # domain bounds
traj_len = 20            # number of iterations
nits = 200                # ensemble repetitions
eps = 0.005              # perturbation amplitude
nbins = 50              # fixed number of histogram bins
index_limits = (3, 11)  # scaling region for fitting

# ---------------------------------------------------------------------------
# 2. Generate Initial Ensemble
# ---------------------------------------------------------------------------

x0s = Sampling_uniform(Ns, xmin, xmax)
params = (a,)

# ---------------------------------------------------------------------------
# 3. Compute Γ, log-distance evolution, and time array
# ---------------------------------------------------------------------------

Gamma_val, ln_avg_dist, t_arr = Gamma_SDE_1D_additive(
    Evolution_rule=tent_map,
    params=params,
    x0s=x0s,
    traj_len=traj_len,
    nits=nits,
    eps=eps,
    Bins_rule="Custom",
    custom_bins=nbins,
    index_limits=index_limits,
    rval=True
)

# ---------------------------------------------------------------------------
# 4. Fit and Plot Results (Equivalent to LLE_Distribution.pdf)
# ---------------------------------------------------------------------------

i0, i1 = index_limits
slope, intercept, r_value, _, _ = linregress(t_arr[i0:i1], ln_avg_dist[i0:i1])

plt.figure(figsize=(5, 3))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.scatter(t_arr, ln_avg_dist, color='gray', alpha=0.5, label=r'$\langle \log(d_i/d_0) \rangle$')
plt.plot(t_arr[i0:i1],
         slope * t_arr[i0:i1] + intercept,
         'g--',
         linewidth=2,
         label=fr'$\Gamma = {slope:.3f}$')

plt.axvline(i0, color='purple', linestyle=':')
plt.axvline(i1, color='purple', linestyle=':')
plt.xlabel(r'$i$', fontsize=16)
plt.ylabel(r'$\langle \log(d_i/d_0) \rangle$', fontsize=16)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# Print summary
print(f"Γ (Gamma) = {slope:.4f},  R² = {r_value**2:.4f}")
