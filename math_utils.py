
import numpy as np

# reuse old gauss quadrature function from Project 1

def orbit_function(nu, orbital_params):
    """"
    This function defines the integrand of the polynomial f(nu, p, e , miu). It's integral provides the
    finite delta t for a change in nu.

    orbital_params = [P, e, miu]
    """
    p,e,miu = orbital_params
    return np.sqrt((p**3)/miu)*(1/(1+e*np.cos(nu))**2)

def GaussQuadApprox(function, nu_range, orbital_params, gauss_pts, method):
    """
    This function utilizes Legendre-Gauss quadrature to estimate an integral:

    deltaT = integral(nu1, nu2) [f(x)] dx
    
    It takes the following inputs: 

    function -> lambda func
    nu_arr = [nu(t1), nu(t2)]
    orbital_params = [p, e, miu]
    gauss_pts = int

    """

    # prepare and destructure all vars
    sum = 0
    nodes, weights = np.polynomial.legendre.leggauss(gauss_pts)
    nu1, nu2 = nu_range

    # offsett nodes since we will be integrating over an arbitrary [a,b] {nu1, nu2} instead of [-1,1]
    corrected_nodes = [((nu2 - nu1)/2)*currentNode + (nu1 + nu2)/2 for currentNode in nodes ]

    if (method.lower() == 'loop'):
        for i in range(gauss_pts): 
            currentProduct = weights[i]*function(corrected_nodes[i], orbital_params)
            sum += currentProduct

        # offset sum by our integration range
        newSum = ((nu2-nu1)/2)*sum
        return newSum
    
    else: # default method, not loop
        weights = np.array(weights)
        function_pts = np.array([function(correctedNode, orbital_params) for correctedNode in corrected_nodes])
        return ((nu2-nu1)/2) * np.dot(weights, function_pts)

## More functions from project 1

def getOrbitalBases(r_vec, v_vec, miu=398600):
    h_vec = np.cross(r_vec, v_vec)
    e_vec = ( (np.cross(v_vec, h_vec)/miu) - (r_vec/np.linalg.norm(r_vec)) )
    return h_vec, e_vec

def findOrbitalParameters(r_vec, v_vec, miu=398600):
    """
    This function takes the position and velocity vectors of an object, and returns
    the six classical orbital elements [a, e, O, i, w, nu] (angles in radians, nu at r_vec),
    plus the semi-latus rectum p, period T, periapsis and apoapsis radii, and the
    eccentricity and angular momentum vectors

    return shape => {elements: [a, e, O, i, w, nu], p, T, rp, ra, e_vec, h_vec}
    """
    #start with eccentricity array, calculate specific angular momentum first
    h_vec, e_arr = getOrbitalBases(r_vec, v_vec, miu)
    h = np.linalg.norm(h_vec)
    e = np.linalg.norm(e_arr)

    #Calculate semi-latus rectum, use SLR to find semi-major axis
    P = (h**2)/miu
    a = P/(1-e**2)

    #now find orbital period
    T = 2*np.pi*np.sqrt((a**3)/miu)

    #periapsis and apoapsis radii (nu = 0 and nu = pi)
    rp = P/(1+e)
    ra = P/(1-e)

    # node vector points to the ascending node, n = K x h
    n_vec = np.cross([0, 0, 1], h_vec)

    i = np.arccos(h_vec[2]/h)

    O = np.arctan2(n_vec[1], n_vec[0])
    if O < 0:  O += 2*np.pi

    w = np.arctan2( np.dot( np.cross(n_vec, e_arr), h_vec ), h*np.dot(n_vec, e_arr) )
    if w < 0:  w += 2*np.pi

    nu = findInitialNu(r_vec, v_vec, miu)

    return {'elements': [a, e, O, i, w, nu],
            'p': P, 'T': T, 'rp': rp, 'ra': ra, 'e_vec': e_arr, 'h_vec': h_vec}

def findInitialNu(r_vec, v_vec, miu=398600):
    """
    Returns the true anomaly in [0, 2pi) at the position r_vec, using the four-quadrant inverse tangent
    """
    h_vec, e_vec = getOrbitalBases(r_vec, v_vec, miu)
    nu = np.arctan2( np.dot( np.cross(e_vec, r_vec), h_vec ), np.linalg.norm(h_vec)*np.dot(e_vec, r_vec) )
    if nu < 0:  nu += 2*np.pi
    return nu

def newton_root(f, f_prime, initial_guess, iters):
    
    x = initial_guess
    x_iters = []
    
    for _ in range(iters):
        x -= f(x) / f_prime(x)
        x_iters.append(x)


    return {'final_value': x, 'iteration_values': x_iters}