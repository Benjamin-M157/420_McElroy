# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
<Name> Benjamin McElroy
<Class> MTH 420 Modern Methods Applied MATh
<Date> Friday may 30th
"""

import numpy as np
from scipy import stats
from scipy import linalg as la
from matplotlib import pyplot as plt


# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """

    points = np.random.uniform(-1, 1, size=(N, n))

    norms = np.linalg.norm(points, axis=1)
    
    inside = np.sum(norms < 1)

    cube_volume = 2 ** n
    
    ball_volume_estimate = (inside / N) * cube_volume
    
    return ball_volume_estimate

if __name__ == "__main__":
    for dim in [2, 3, 4]:
        est_vol = ball_volume(dim)
        print(f"Estimated volume of U{dim}: {est_vol:.5f}")
        


# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    
    x = np.random.uniform(a, b, N)
    
    fx = f(x)
    
    estimate = (b - a) * np.mean(fx)
    
    return estimate

if __name__ == "__main__":
    from math import log, pi

    f1 = lambda x: x**2
    f2 = lambda x: np.sin(x)
    f3 = lambda x: 1 / x
    f4 = lambda x: np.abs(np.sin(10*x) * np.cos(10*x) + np.sqrt(np.abs(x)) * np.sin(3*x))

    # Test cases
    print("∫₋₄² x² dx ≈", mc_integrate1d(f1, -4, 2))         
    print("∫₋₂π²π sin(x) dx ≈", mc_integrate1d(f2, -2*pi, 2*pi))  
    print("∫₁¹⁰ 1/x dx ≈", mc_integrate1d(f3, 1, 10))      
    print("∫₁⁵ complex dx ≈", mc_integrate1d(f4, 1, 5))     
    
    



# Problem 3
def mc_integrate(f, a_bounds, b_bounds, N=10**4):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    
    a_bounds = np.array(a_bounds)
    b_bounds = np.array(b_bounds)
    dim = len(a_bounds)
    
    volume = np.prod(b_bounds - a_bounds)
    
    samples = np.random.rand(N, dim)
    scaled_samples = a_bounds + samples * (b_bounds - a_bounds)
    
    values = np.array([f(x) for x in scaled_samples])
    
    return volume * np.mean(values)

if __name__ == "__main__":
    f1 = lambda x: x[0]**2 + x[1]**2
    result1 = mc_integrate(f1, [0, 0], [1, 1])
    print(f"Test 1 (Expected ≈ 0.6667): {result1:.5f}")

    f2 = lambda x: 3*x[0] - 4*x[1] + x[1]**2
    result2 = mc_integrate(f2, [-2, 1], [1, 3])
    print(f"Test 2 (Expected ≈ 54): {result2:.5f}")

    f3 = lambda x: x[0] + x[1] - x[3] * x[2]**2
    result3 = mc_integrate(f3, [-1, -2, -3, -4], [1, 2, 3, 4])
    print(f"Test 3 (Expected ≈ 0): {result3:.5f}")



# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    raise NotImplementedError("Problem 4 Incomplete")
