import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def earthSphere(ax, R_e, image_path='earth.jpg', n=100):
    """
    Python version of earthSphere.m: draws an Earth-sized sphere on a 3D axis,
    textured with earth.jpg (equirectangular) if it exists, plain otherwise
    """
    lon = np.linspace(-np.pi, np.pi, 2*n)
    lat = np.linspace(np.pi/2, -np.pi/2, n)
    lon, lat = np.meshgrid(lon, lat)

    x = R_e*np.cos(lat)*np.cos(lon)
    y = R_e*np.cos(lat)*np.sin(lon)
    z = R_e*np.sin(lat)

    if os.path.exists(image_path):
        img = plt.imread(image_path)
        if img.dtype == np.uint8:  img = img/255

        # sample the image on the sphere grid (rows -> lat, cols -> lon)
        rows = np.linspace(0, img.shape[0] - 1, n).astype(int)
        cols = np.linspace(0, img.shape[1] - 1, 2*n).astype(int)
        colors = img[np.ix_(rows, cols)][..., :3]

        ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, shade=False)
    else:
        ax.plot_surface(x, y, z, color='tab:blue', alpha=0.6, linewidth=0)

def mercatorDisplay(lon_arr, lat_arr, map_path='worldmap.mat', image_path='earth.jpg'):
    """
    Python version of mercatorDisplay.m: plots the groundtrack (lon, lat in radians) over an equirectangular
    Earth image if it exists, otherwise over the coastlines in worldmap.mat
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if os.path.exists(image_path):
        # same as image('CData',earth,'XData',[-180 180],'YData',[90 -90])
        ax.imshow(plt.imread(image_path), extent=[-180, 180, -90, 90])
    else:
        world = loadmat(map_path)
        ax.plot(world['lon'], world['lat'], '.', markersize=3, color='tab:blue')

    ax.plot(np.rad2deg(lon_arr), np.rad2deg(lat_arr), '*', markersize=4, color='red')

    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_box_aspect(1/2)
    ax.set_xlabel('Earth-relative longitude (deg)')
    ax.set_ylabel('Geocentric latitude (deg)')

    return fig, ax
