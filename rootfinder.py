from math_utils import *
import numpy as np

delta_nu = np.deg2rad(5)  # initial guess increment
iter_num = 20

# f is the integrand! 

def rootFinder(f, t_0, nu_0, t, p, e, mu, N):
    """
    This function computes the value of F(nu) that satisfies the given time delta, 
    thus returning the value of the ellapsed true anomaly for that time range.

    f = integrand of F(nu)
    t_0 = start time
    nu_0 = start true anomaly (rad)
    t = terminus time
    [p,e,mu] = orbital parameters
    N = gauss points 

    """

    orbital_params = [p, e, mu]

    # set up the functions, G(nu) = F(nu) - (t - t_0), F(nu) approximated with Gauss
    G_nu = lambda x: GaussQuadApprox(
                           function=f,
                           nu_range=[nu_0, x],
                           orbital_params=orbital_params,
                           gauss_pts=N,
                           method='loop'
                           ) - (t - t_0)

    G_nu_prime = lambda x: f(x, orbital_params)

    # call helpers to do root finding
    res = newton_root(
                f=G_nu,
                f_prime=G_nu_prime,
                initial_guess=nu_0 + delta_nu,
                iters=iter_num
                )

    return res['final_value']