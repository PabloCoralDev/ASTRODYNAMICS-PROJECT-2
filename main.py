import os
import numpy as np
import matplotlib.pyplot as plt

from math_utils import *
from computeNu import computeNu
from computeRandV import computeRandV
from eci2ecef import eci2ecef
from ecef2LonLat import ecef2LonLat
from plotting import earthSphere, mercatorDisplay

## write out all necessary vars

# 1. position and velocity
r_0 = np.array([-1217.39430415697, -3091.41210822807, -6173.40732877317]) # km
v_0 = np.array([9.88635815507896, -0.446121737099303, -0.890884522967222]) # km/s

# 2. Physical Constants
miu = 398600 # km^3/s^2
O_e = 7.2921159e-5 # rad/s, Earth rotation rate
R_e = 6378.1363 # km, Earth radius

# 3. Mathematical constants / offsets / points
delta_t_mins = 5
delta_t = delta_t_mins*60  # s
num_periods = 2
gauss_pts = 20

# givens
lon_band_1 = [80, 100] # deg
lon_band_2 = [-100, -80] # deg

view_az, view_el = 49.5, 22.8  # MATLAB's view(49.5, 22.8)

## set up output dir filepath + image file path
fig_dir = 'figures'
os.makedirs(fig_dir, exist_ok=True)
earth_img_path = os.path.join(fig_dir, 'equirect_map_NASA.jpg')  # NASA equirectangular Earth image

## Task 1-2 : get true anomaly at every point in orbit -------------------------- #

res = findOrbitalParameters(r_0, v_0, miu)

a, e, O, i, w, nu_0= res['elements']
p, T = res['p'], res['T']
rp, ra = res['rp'], res['ra']

# get the time from apoapsis (t = 0) to the current nu_0
t_0 = GaussQuadApprox(orbit_function, [np.pi, nu_0], [p, e, miu], gauss_pts, 'loop')

# find total time on a period basis, then build the t_array that will be used alongside nu to draw the orbit
t_n = t_0 + num_periods*T
t_arr = np.arange(t_0, t_n + delta_t/2, delta_t)

# use our rootfinder script in 'batch' to find nu's corresponding to t_arr
nu_arr = computeNu(t_arr, r_0, v_0, miu)

## ------------------------------ END OF TASK 1 --------------------------------- ##

## Task 2: ECI positions and velocities throughout orbit ------------------------ #
r_arr, v_arr = computeRandV(nu_arr, a, e, O, i, w, miu)

## ------------------------------ END OF TASK 2 --------------------------------- ##

## Task 3: Use rotations to find all r in ECEF. Find lat long with given eqns --- #
R_ecef = eci2ecef(t_arr, r_arr, O_e)
lon_arr, lat_arr = ecef2LonLat(R_ecef)

print("\nOrbital elements and properties")
print(f"\ta\t= {a:.4f} km")
print(f"\te\t= {e:.6f}")
print(f"\tO\t= {np.rad2deg(O):.4f} deg")
print(f"\ti\t= {np.rad2deg(i):.4f} deg")
print(f"\tw\t= {np.rad2deg(w):.4f} deg")
print(f"\tnu_0\t= {np.rad2deg(nu_0):.4f} deg")
print(f"\tT\t= {T:.4f} s")
print(f"\trp\t= {rp:.4f} km")
print(f"\tra\t= {ra:.4f} km")
print(f"\tt_0\t= {t_0:.4f} s")

## ------------------------------ END OF TASK 3 --------------------------------- ##


## ------------------------------ TASK 4 ---------------------------------------- #

## Task 4-1: Plot the orbit in 3D ----------------------------------------------- #

# scaffold figure and axes
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection='3d')

# render the earth
earthSphere(ax, R_e, image_path=earth_img_path)

# plot the orbit
ax.plot(r_arr[:, 0], r_arr[:, 1], r_arr[:, 2], linewidth=2, color='tab:red')

# housekeeping stuff so plot looks good + labels and saving
ax.set_aspect('equal')
ax.view_init(elev=view_el, azim=view_az - 90)  # substract 90 to azimuth since MATLAB measures from -y and matplotlib from +x
ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_zlabel('z (km)')

fig.savefig(os.path.join(fig_dir, 'orbit3d.png'), dpi=200, bbox_inches='tight')

## Task 4-2: plot groundtrack ---------------------------------------------------------- #
fig, ax = mercatorDisplay(lon_arr, lat_arr, image_path=earth_img_path)
fig.savefig(os.path.join(fig_dir, 'groundtrack.png'), dpi=200, bbox_inches='tight')

## Task 4-3: find time spent in each longitude band ------------------------------------ #
lon_deg_arr = np.rad2deg(lon_arr)
lat_deg_arr = np.rad2deg(lat_arr)
r_mag_arr = np.linalg.norm(r_arr, axis=1)

band_locns = ['Turukhansky District, Russia', 'Kivallic Region, Canada'] # found thru gps-coodinates.net

for band, locn in zip([lon_band_1, lon_band_2], band_locns): # make a dir with band + locn

    # lon_band_n -> [start, end] ; make binary array and add instances, each instance counts for a delta T
    in_band_arr = (lon_deg_arr >= band[0]) & (lon_deg_arr <= band[1]) # returns [0,0,0,1,1,1,1,0,0,0] etc...
    
    # find indices of in band points (can use to go thru other arrays)
    idx_arr = np.where(in_band_arr)[0] # indices of pts inside the band from OG array --> need [0] bc .where returns ([dim1], [dim2])... and we have ([dim1])

    # mean radius of band determines proximity to rp or ra
    r_mean = np.mean(r_mag_arr[idx_arr]) 
    near = 'apoapsis' if abs(r_mean - ra) < abs(r_mean - rp) else 'periapsis' 

    in_band_latitudes = lat_deg_arr[idx_arr]
    max_lat_short_idx = np.argmax(in_band_latitudes) # get position of max lat in in-band latitutdes
    j_max = idx_arr[max_lat_short_idx] # lookup in idx_arr -> [3,4,5,6] means band starts at pos #3; if short max is at band idx 2, then it will be at lat/long arr pos 5
    
    # prepare results
    total_time_hr = np.sum(in_band_arr)*delta_t/3600
    max_long = lon_deg_arr[j_max]
    max_lat = lat_deg_arr[j_max]

    print(f"\nLongitude band [{band[0]}, {band[1]}] deg")
    print(f"\ttime spent\t= {total_time_hr:.4f} hr")
    print(f"\tsegment near\t= {near}")
    print(f"\tlong, lat = ({max_long:.4f}, {max_lat:.4f}) deg")
    print(f"\tlocation\t= {locn}") 

plt.show()

## ------------------------------ END OF TASK 4 --------------------------------- ##
