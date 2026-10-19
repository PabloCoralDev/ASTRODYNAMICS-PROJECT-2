from math_utils import *
from rootfinder import rootFinder
import numpy as np


def computeNu(t_arr, r_0, v_0, miu, gauss_pts=20):
    """
    This function computes an array of ellapsed true anomaly values given a 
    starting r, v and an array of time values.

    """

    # destruct from helper
    res = findOrbitalParameters(r_0, v_0, miu)
    p = res['p']
    _, e, _, _, _, nu_0 = res['elements']

    # scaffold output array (same size as t) and set IC
    nu_arr = np.zeros(len(t_arr))
    nu_arr[0] = nu_0

    # populate array starting from second position [1] and use previous values as inputs
    for i in range(1, len(t_arr)):
        nu_arr[i] = rootFinder(orbit_function, t_arr[i-1], nu_arr[i-1], t_arr[i], p, e, miu, gauss_pts)

    return nu_arr
