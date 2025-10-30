import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import ot
from scipy.stats import linregress
from tqdm import tqdm
from joblib import Parallel, delayed

# ---------------------------------------------------------------------------
# Sampling Functions
# ---------------------------------------------------------------------------

def Sampling_uniform(Ns: int, xmin: float, xmax: float) -> np.ndarray:
    """Generate samples from a uniform distribution.

    Parameters
    ----------
    Ns : int
        Number of samples.
    xmin : float
        Minimum value of the uniform distribution.
    xmax : float
        Maximum value of the uniform distribution.

    Returns
    -------
    np.ndarray
        Samples from the uniform distribution.
    """
    return np.random.uniform(xmin, xmax, Ns)


def Sampling_normal(Ns: int, loc: float, scl: float) -> np.ndarray:
    """Generate samples from a normal distribution.

    Parameters
    ----------
    Ns : int
        Number of samples.
    loc : float
        Mean of the normal distribution.
    scl : float
        Standard deviation of the normal distribution.

    Returns
    -------
    np.ndarray
        Samples from the normal distribution.
    """
    return np.random.normal(loc, scl, Ns)

# ---------------------------------------------------------------------------
# Distance and Binning Functions
# ---------------------------------------------------------------------------

def EMD_histogram_1D(x0s: np.ndarray, xNs: np.ndarray, nbins: int) -> float:
    """Compute Earth Mover's Distance (Wasserstein-1) between two histograms.

    Parameters
    ----------
    x0s : np.ndarray
        Samples from the first distribution.
    xNs : np.ndarray
        Samples from the second distribution.
    nbins : int
        Number of bins for the histogram.

    Returns
    -------
    float
        Earth Mover's Distance between the two distributions.
    """
    hist, bin_edges = np.histogram(x0s, bins=nbins)
    P0 = hist / np.sum(hist)
    xvals0 = (bin_edges[:-1] + bin_edges[1:]) / 2

    hist, bin_edges = np.histogram(xNs, bins=nbins)
    PN = hist / np.sum(hist)
    xvalsN = (bin_edges[:-1] + bin_edges[1:]) / 2

    M = ot.dist(xvals0.reshape((-1, 1)), xvalsN.reshape((-1, 1)), 'euclidean')
    return ot.emd2(P0, PN, M)


def compute_bins(data: np.ndarray, method: str = "Sqrt", custom_bins: int = None) -> int:
    """Compute number of histogram bins using standard rules.

    Parameters
    ----------
    data : np.ndarray
        Dataset for which to determine the number of bins.
    method : str, optional
        Method for computing number of bins ('Sturges', 'Sqrt', 'Rice', or 'Custom').
    custom_bins : int, optional
        Custom number of bins if method is 'Custom'.

    Returns
    -------
    int
        Number of bins for histogram construction.
    """
    n = len(data)
    if method == "Sturges":
        return int(np.ceil(np.log2(n) + 1))
    if method == "Sqrt":
        return int(np.ceil(np.sqrt(n)))
    if method == "Rice":
        return int(np.ceil(2 * n ** (1 / 3)))
    if method == "Custom" and custom_bins is not None:
        return custom_bins
    raise ValueError("Invalid method or missing custom_bins for 'Custom'.")

# ---------------------------------------------------------------------------
# Evolution and plotting functions
# ---------------------------------------------------------------------------
def Plot_histogram(Probabilities,bin_edges,bin_widths):
    """plots a histogram

    Parameters
    ----------
    Probabilities : array
        probabilities of the bins
    bin_edges : array
        bin edges for the histogram
    bin_widths : array
        widths of the bins
    """
    plt.figure(figsize=(2, 2))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.bar(bin_edges[:-1],Probabilities, width=bin_widths, edgecolor="black", align="edge")
    plt.xlabel(r'$x$')
    plt.ylabel(r'$P(x)$')
    #plt.legend()
    plt.show()
    return

def Evolve_1D_map(Evolution_rule,params,x0s,Bins_rule='Sqrt',custom_bins=None,T=300,show_plt = True):
    """returns the samples from the distribution after a number of iterations   

    Parameters
    ----------
    Evolution_rule : function
        map function
    params : array
        parameters of the map function
    x0s : array
        samples from the original distribution
    Bins_rule : str
        method to compute number of bins
    custom_bins : int
        number of bins
    T : int
        number of iterations
    show_plt : bool 
        show the plot of the histogram

    Returns
    -------
    x0s : array
        samples from the distribution after T iterations
    """ 
    #1. Compute number of samples 
    Ns = len(x0s)
    #2. Using number of samples decide on number of bins
    nbins= compute_bins(x0s,Bins_rule, custom_bins)
    for i in range(T):
        x0s=Evolution_rule(x0s, *params)
    # Calculate the bin edges widths and centers
    hist, bin_edges = np.histogram(x0s, bins=nbins)
    bin_widths = [bin_edges[i+1] - bin_edges[i] for i in range(len(bin_edges) - 1)]
    # 1. Histogram of fiducial trajectory
    P0 = hist/Ns
    if show_plt == True:
        Plot_histogram(P0,bin_edges,bin_widths)
    return x0s

# ---------------------------------------------------------------------------
# Perturbation Functions
# ---------------------------------------------------------------------------

def Perturbation_additive_1D(x0s: np.ndarray, nbins: int, eps: float) -> np.ndarray:
    """Generate a perturbed distribution by adding Gaussian noise to histogram probabilities.

    Parameters
    ----------
    x0s : np.ndarray
        Samples from the original distribution.
    nbins : int
        Number of bins for the histogram.
    eps : float
        Standard deviation of the perturbation.

    Returns
    -------
    np.ndarray
        Samples from the perturbed distribution.
    """
    ns = len(x0s)
    hist, bin_edges = np.histogram(x0s, bins=nbins)
    P0 = hist / ns

    perturb = np.random.normal(0, eps, size=hist.shape)
    Pn = np.maximum(P0 + perturb, 0)
    Pn /= np.sum(Pn)

    bin_indices = np.random.choice(len(Pn), size=ns, p=Pn)
    return np.random.uniform(low=bin_edges[bin_indices], high=bin_edges[bin_indices + 1])

# ---------------------------------------------------------------------------
# Gamma (Γ) Computation Functions
# ---------------------------------------------------------------------------

def Gamma_single_iteration(Evolution_rule: callable, params: tuple, x0s: np.ndarray, xNs: np.ndarray, nbins: int, traj_len: int):
    """Compute normalized EMD evolution between perturbed and unperturbed ensembles.

    Parameters
    ----------
    Evolution_rule : callable
        Function defining the system's evolution rule.
    params : tuple
        Parameters for the map function.
    x0s : np.ndarray
        Samples from the unperturbed ensemble.
    xNs : np.ndarray
        Samples from the perturbed ensemble.
    nbins : int
        Number of histogram bins.
    traj_len : int
        Number of time steps for evolution.

    Returns
    -------
    np.ndarray
        EMD values as a function of time.
    np.ndarray
        Updated unperturbed samples.
    np.ndarray
        Updated perturbed samples.
    """
    D_t = np.zeros(traj_len)
    D0 = EMD_histogram_1D(x0s, xNs, nbins)
    for i in range(traj_len):
        D_t[i] = EMD_histogram_1D(x0s, xNs, nbins) / D0
        x0s = Evolution_rule(x0s, *params)
        xNs = Evolution_rule(xNs, *params)
    return D_t, x0s, xNs


def Gamma_ln_avg_distance(D_t_arr: np.ndarray, traj_len: int) -> np.ndarray:
    """Compute ensemble-averaged logarithmic separation ⟨log(d(t))⟩.

    Parameters
    ----------
    D_t_arr : np.ndarray
        Matrix of EMD distances over time (shape: [n_iter, traj_len]).
    traj_len : int
        Number of time steps.

    Returns
    -------
    np.ndarray
        Average logarithmic separation for each time step.
    """
    ln_avg = np.zeros(traj_len)
    for k in range(traj_len):
        vals = D_t_arr[:, k]
        nonzero = vals[vals > 0]
        ln_avg[k] = np.mean(np.log(nonzero)) if len(nonzero) > 0 else -np.inf
    return ln_avg


def Gamma_SDE_1D_additive(Evolution_rule: callable, params: tuple, x0s: np.ndarray, traj_len: int, nits: int, eps: float = 0.001, Bins_rule: str = 'Sqrt', custom_bins: int = None, index_limits: tuple = None, rval: bool = False):
    """Compute Γ (average rate of ensemble separation) for a stochastic 1D map.

    Parameters
    ----------
    Evolution_rule : callable
        Evolution map defining the system dynamics.
    params : tuple
        Parameters for the evolution rule.
    x0s : np.ndarray
        Initial ensemble samples.
    traj_len : int
        Number of time steps for each trajectory.
    nits : int
        Number of iterations used to estimate ensemble average.
    eps : float, optional
        Amplitude of additive perturbation noise.
    Bins_rule : str, optional
        Rule to determine number of histogram bins ('Sqrt', 'Sturges', etc.).
    custom_bins : int, optional
        Custom bin count (used if Bins_rule='Custom').
    index_limits : tuple, optional
        Tuple of (start, end) indices for fitting scaling region.
    rval : bool, optional
        If True, returns Γ, log distances, and time array.

    Returns
    -------
    float or tuple
        Γ value or tuple (Γ, log-distances, time array) if rval=True.
    """
    nbins = compute_bins(x0s, Bins_rule, custom_bins)
    xNs = Perturbation_additive_1D(x0s, nbins, eps)

    D_t_arr = np.zeros((nits, traj_len))
    for i in range(nits):
        D_t_arr[i], x0s, xNs = Gamma_single_iteration(Evolution_rule, params, x0s, xNs, nbins, traj_len)
        xNs = Perturbation_additive_1D(x0s, nbins, eps)

    ln_avg = Gamma_ln_avg_distance(D_t_arr, traj_len)
    t_arr = np.arange(traj_len)

    if index_limits:
        i0, i1 = index_limits
        slope, _, r, _, _ = linregress(t_arr[i0:i1], ln_avg[i0:i1])
    else:
        slope, _, r, _, _ = linregress(t_arr, ln_avg)

    return (slope, ln_avg, t_arr) if rval else slope

# ---------------------------------------------------------------------------
# Parallel Computation and Visualization
# ---------------------------------------------------------------------------

def compute_Gamma_for_eps(Evolution_rule: callable, params: tuple, Ns: int, xmin: float, xmax: float, traj_len: int, nits: int, eps_val: float, nbins: int, ilimits: tuple):
    """Compute Γ for a given perturbation amplitude ε.

    Parameters
    ----------
    Evolution_rule : callable
        Evolution map.
    params : tuple
        Parameters for the map.
    Ns : int
        Number of ensemble samples.
    xmin : float
        Minimum initial value.
    xmax : float
        Maximum initial value.
    traj_len : int
        Number of iterations.
    nits : int
        Number of ensemble repetitions.
    eps_val : float
        Perturbation amplitude.
    nbins : int
        Number of bins for histogram computation.
    ilimits : tuple
        Scaling region index limits.

    Returns
    -------
    tuple
        (eps_val, {"Gamma": computed Γ value}).
    """
    x0s = Sampling_uniform(Ns, xmin, xmax)
    Gamma_val = Gamma_SDE_1D_additive(Evolution_rule, params, x0s, traj_len, nits, eps_val, 'Custom', nbins, ilimits)
    return eps_val, {"Gamma": Gamma_val}


def parallel_Gamma_over_eps(eps_array: np.ndarray, Evolution_rule: callable, params: tuple, Ns: int, xmin: float, xmax: float, traj_len: int, nits: int, nbins: int, ilimits: tuple, n_jobs: int = -1) -> dict:
    """Compute Γ across multiple ε values in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_Gamma_for_eps)(Evolution_rule, params, Ns, xmin, xmax, traj_len, nits, eps, nbins, ilimits)
        for eps in tqdm(eps_array, desc="Gamma over epsilon")
    )
    return {eps: res for eps, res in results}


def Gamma_vs_perturbation_vs_samples(Ns_array: np.ndarray, Evolution_rule: callable, params: tuple, xmin: float, xmax: float, traj_len: int, nits: int, epsMin: float, epsMax: float, epsPts: int, nbins: int, index_limits: tuple, fname: str = None) -> dict:
    """Compute Γ variation with perturbation ε and sample size Nₛ.

    Parameters
    ----------
    Ns_array : np.ndarray
        Array of sample sizes.
    Evolution_rule : callable
        Evolution map.
    params : tuple
        Parameters for the map.
    xmin : float
        Minimum initialization value.
    xmax : float
        Maximum initialization value.
    traj_len : int
        Number of iterations per trajectory.
    nits : int
        Number of ensemble repetitions.
    epsMin : float
        Minimum perturbation amplitude.
    epsMax : float
        Maximum perturbation amplitude.
    epsPts : int
        Number of perturbation points.
    nbins : int
        Number of bins for histogram.
    index_limits : tuple
        Scaling region index limits.
    fname : str, optional
        Output filename to save results.

    Returns
    -------
    dict
        Dictionary containing Γ values vs ε for each Nₛ.
    """
    eps_array = np.linspace(epsMin, epsMax, epsPts)
    data = {}
    for Ns in tqdm(Ns_array, desc="Outer loop over Ns"):
        result_dict = parallel_Gamma_over_eps(eps_array, Evolution_rule, params, Ns, xmin, xmax, traj_len, nits, nbins, index_limits)
        sorted_eps = sorted(result_dict.keys())
        G_array = np.array([result_dict[eps]["Gamma"] for eps in sorted_eps])
        data[Ns] = {"eps": np.array(sorted_eps), "Gamma": G_array}
    if fname:
        np.save(fname, data)
    return data


def Plot_Gamma_vs_perturbation(fn: str, NsArr: np.ndarray):
    """Plot Γ as a function of perturbation amplitude ε for different sample sizes.

    Parameters
    ----------
    fn : str
        Filename of saved Γ data.
    NsArr : np.ndarray
        Array of sample sizes used in computation.
    """
    data = np.load(fn, allow_pickle=True).item()
    cmap = cm.rainbow
    norm = Normalize(vmin=np.min(NsArr), vmax=np.max(NsArr))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)

    fig, ax = plt.subplots(figsize=(6, 3))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    for Ns in NsArr:
        color = cmap(norm(Ns))
        eps_vals = data[Ns]["eps"]
        G_vals = data[Ns]["Gamma"]
        ax.plot(eps_vals, G_vals, color=color)

    ax.set_xlabel(r'$\\epsilon$', fontsize=16)
    ax.set_ylabel(r'$\\Gamma$ (Average rate of separation)', fontsize=14)
    cbar = plt.colorbar(sm, ax=ax, pad=0.02, shrink=0.9)
    cbar.set_label(r'$N_s$', fontsize=14)
    plt.tight_layout()
    plt.show()

    return fig, ax
