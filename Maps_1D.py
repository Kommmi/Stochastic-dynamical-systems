import numpy as np


def reflect_bound(x, xmin, xmax):
    x = np.where(x < xmin, 2 * xmin - x, x)
    x = np.where(x > xmax, 2 * xmax - x, x)
    return x

def cusp_map(x,b,a):
    """
    Cusp map function.

    Parameters:
    x (float) : Current state.
    a (float): Map parameter.
    b (float): Map parameter.

    Returns:
    float: Next state.
    """
    return a*(1-np.abs(1-1.9999*x)**b)

def noisy_cusp_map_scalar(x0,b,a,noise_std=0.0001,bounds='reflect'):
    """
    Generates a trajectory for a noisy logistic map.
    
    Parameters:
    x0 (float): The initial condition.
    a (float): The map parameter.
    b (float): The map parameter.
    noise_std (float): Standard deviation of the Gaussian noise decides the noise level.
    
    Returns:
    x_new(float): Next state.
    """
    noise = np.random.normal(0, noise_std)
    x_new = cusp_map(x0, b,a) + noise
    if bounds=='reflect':
        x_new = reflect_bound(x_new, 0, 1)
    else:
        x_new = np.clip(x_new, 0, 1)
    return x_new

def noisy_cusp_map(x0,b,a,noise_std=0.0001,bounds='reflect'):
    """
    Generates a trajectory for a noisy logistic map.
    
    Parameters:
    x0 (float): The initial condition.
    a (float): The map parameter.
    b (float): The map parameter.
    noise_std (float): Standard deviation of the Gaussian noise decides the noise level.
    
    Returns:
    x_new(float): Next state.
    """
    noise = np.random.normal(0, noise_std, size=x0.shape[0])
    x_new = cusp_map(x0, b,a) + noise
    if bounds=='reflect':
        x_new = reflect_bound(x_new, 0, 1)
    else:
        x_new = np.clip(x_new, 0, 1)
    return x_new

# Example of another map function: Tent map
def tent_map(x, mu):
    """
    Tent map function.

    Parameters:
    x (float) : Current state.
    mu (float): Map parameter.

    Returns:
    float: Next state.
    """
    return np.where(x < 0.5, mu * x, mu * (1 - x))


def noisy_tent_map_scalar(x0,mu,noise_std=0.0001, xmin=0, xmax=1, bounds='reflect'):
    """
    Generates a trajectory for a noisy logistic map.
    
    Parameters:
    x0 (float): The initial condition.
    mu (float): The map parameter.
    noise_std (float): Standard deviation of the Gaussian noise decides the noise level.
    
    Returns:
    x_new(float): Next state.
    """
    noise = np.random.normal(0, noise_std)
    x_new = tent_map(x0, mu) + noise
    if bounds=='reflect':
        x_new = reflect_bound(x_new, xmin, xmax)
    else:
        x_new = np.clip(x_new, xmin, xmax)
    return x_new

def noisy_tent_map(x0,mu,noise_std=0.0001,xmin=0,xmax=1,bounds='reflect'):
    """
    Generates a trajectory for a noisy logistic map.
    
    Parameters:
    x0 (float): The initial condition.
    mu (float): The map parameter.
    noise_std (float): Standard deviation of the Gaussian noise decides the noise level.
    
    Returns:
    x_new(float): Next state.
    """
    noise = np.random.normal(0, noise_std,size=x0.shape[0])
    x_new = tent_map(x0, mu) + noise
    if bounds=='reflect':
        x_new = reflect_bound(x_new, xmin, xmax)
    else:
        x_new = np.clip(x_new, xmin, xmax)
    return x_new


def Chain_climbing_sine_map(x, a):
    """
    Chain climbing sine map function.

    Parameters:
    x (float) : Current state.
    a (float): Map parameter.

    Returns:
    float: Next state.
    """
    return a*np.sin(2*np.pi * x) +  x

def noisy_chain_climbing_sine_map_scalar(x0, a, noise_std=0.0001, xmin=-1, xmax=2,bounds='reflect'):
    """
    Generates a trajectory for a noisy chain climbing sine map with reflective boundaries.
    
    Parameters:
    x0 (float): Initial condition.
    a (float): Map parameter.
    noise_std (float): Standard deviation of Gaussian noise.
    
    Returns:
    float: Next state with reflecting boundaries applied.
    """
    noise = np.random.normal(0, noise_std)
    x_new = Chain_climbing_sine_map(x0, a) + noise
    if bounds=='reflect':
        x_new = reflect_bound(x_new, xmin, xmax)
    else:
        x_new = np.clip(x_new, xmin, xmax)
    return x_new

def noisy_chain_climbing_sine_map(x0, a, noise_std=0.0001, xmin=-1, xmax=2,bounds='clip'):
    """
    Generates a trajectory for a noisy chain climbing sine map with reflective boundaries.
    
    Parameters:
    x0 (np.ndarray): Initial condition(s), should be a NumPy array.
    a (float): Map parameter.
    noise_std (float): Standard deviation of Gaussian noise.
    
    Returns:
    np.ndarray: Next state with reflecting boundaries applied.
    """
    noise = np.random.normal(0, noise_std, size=x0.shape[0])
    x_new = Chain_climbing_sine_map(x0, a) + noise
    if bounds=='reflect':
        x_new = reflect_bound(x_new, xmin, xmax)
    else:
        x_new = np.clip(x_new, xmin, xmax)
    return x_new


def heterogenous_sine_map(x, a):
    """
    Heterogeneous sine map function.

    Parameters:
    x (float) : Current state.
    a (float): Map parameter.

    Returns:
    float: Next state.
    """
    return a*np.sin(2*np.pi * x) +  x + np.sin(2*x)

def noisy_heterogenous_sine_map_scalar(x0,a,noise_std=0.0001):
    """
    Generates a trajectory for a noisy heterogeneous sine map.
    
    Parameters:
    x0 (float): The initial condition.
    a (float): The map parameter.
    noise_std (float): Standard deviation of the Gaussian noise decides the noise level.
    
    Returns:
    x_new(float): Next state.
    """
    #print(x0.shape)
    # Convert scalar to array if necessary
    noise = np.random.normal(0, noise_std)
    x_new = heterogenous_sine_map(x0, a) + noise
    return x_new

def noisy_heterogenous_sine_map(x0,a,noise_std=0.0001):
    """
    Generates a trajectory for a noisy heterogeneous sine map.
    
    Parameters:
    x0 (float): The initial condition.
    a (float): The map parameter.
    noise_std (float): Standard deviation of the Gaussian noise decides the noise level.
    
    Returns:
    x_new(float): Next state.
    """
    #print(x0.shape)
    # Convert scalar to array if necessary
    noise = np.random.normal(0, noise_std,size=x0.shape[0])
    x_new = heterogenous_sine_map(x0, a) + noise
    return x_new

def logistic_map(x, r):
    """
    Logistic map function.

    Parameters:
    x (float): Current state.
    r (float): Growth rate parameter.

    Returns:
    float: Next state.
    """
    return r * x * (1 - x)

def noisy_logistic_map_scalar(x0,mu,noise_std=0.0001,bounds='reflect'):
    """
    Generates a trajectory for a noisy logistic map.
    
    Parameters:
    x0 (float): The initial condition.
    mu (float): The map parameter.
    noise_std (float): Standard deviation of the Gaussian noise decides the noise level.
    
    Returns:
    x_new(float): Next state.
    """
    noise = np.random.normal(0, noise_std)
    x_new = logistic_map(x0, mu) + noise
    if bounds=='reflect':
        x_new = reflect_bound(x_new, 0, 1)
    else:
        x_new = np.clip(x_new, 0, 1)
    return x_new

def noisy_logistic_map(x0,mu,noise_std=0.0001,bounds='reflect'):
    """
    Generates a trajectory for a noisy logistic map.
    
    Parameters:
    x0 (float): The initial condition.
    mu (float): The map parameter.
    noise_std (float): Standard deviation of the Gaussian noise decides the noise level.
    
    Returns:
    x_new(float): Next state.
    """
    noise = np.random.normal(0, noise_std,size=x0.shape[0])
    x_new = logistic_map(x0, mu) + noise
    if bounds=='reflect':
        x_new = reflect_bound(x_new, 0, 1)
    else:
        x_new = np.clip(x_new, 0, 1)
    return x_new

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
