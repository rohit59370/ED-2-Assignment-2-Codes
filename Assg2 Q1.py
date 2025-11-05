#!/usr/bin/env python3
"""
E and B field for a moving charge (no acceleration).
2D plots for 3 speeds + 3D field plots (E and B) for all 3 speeds.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots

# -------------------------
# my setup stuff
# -------------------------
q = 1e-9             # charge value
grid_halfwidth = 2.0 # size of the 2D plotting area (in meters)
n_grid = 300         # how fine the grid is (more = slower)

# 3D plotting options
N3D = 12             # number of points per axis (keep this small or it gets messy)
L3D = 0.8            # size of cube around the charge for 3D field
length3d = 0.18      # max arrow length in 3D plots
mask_cutoff = 0.06   # ignore points too close to charge (field goes crazy there)
MAG_MAPPING = 'sqrt' # how to scale arrow lengths ('linear', 'sqrt', 'log')

# 2D plotting settings
QUIVER_SUBSAMPLE = 12   # how many arrows to skip for 2D field plot
QUIVER_WIDTH = 0.004    # thickness of arrows
ARROW_COLOR = "black"
SHOW_PLOTS = True

# -------------------------
# constants
# -------------------------
eps0 = 8.8541878128e-12
c = 299792458.0

# -------------------------
# field formulas
# -------------------------
def compute_E_field(q, vx, X, Y, Z=None):
    # returns the electric field (2D or 3D depending on args)
    beta = vx / c
    beta2 = beta**2

    if Z is None:
        Rx, Ry, Rz = X, Y, np.zeros_like(X)
    else:
        Rx, Ry, Rz = X, Y, Z

    R = np.sqrt(Rx**2 + Ry**2 + Rz**2)
    R_safe = np.where(R == 0.0, 1e-20, R)

    Rperp2 = Ry**2 + Rz**2
    sin2 = Rperp2 / (R_safe**2)
    denom = np.maximum(1.0 - beta2 * sin2, 1e-20)
    prefactor = (q / (4.0 * np.pi * eps0)) * (1.0 - beta2) / (denom ** 1.5)

    Ex = prefactor * (Rx / (R_safe**3))
    Ey = prefactor * (Ry / (R_safe**3))
    Ez = prefactor * (Rz / (R_safe**3))

    if Z is None:
        return Ex, Ey
    else:
        return Ex, Ey, Ez

def compute_B_from_E(vx, Ex, Ey, Ez=None):
    # just v x E / c^2 basically (v along +x)
    if Ez is None:
        Ez = np.zeros_like(Ex)
    Bx = np.zeros_like(Ex)
    By = - (vx * Ez) / (c**2)
    Bz =   (vx * Ey) / (c**2)
    return Bx, By, Bz

def map_magnitude_to_unit(mag, mapping='sqrt'):
    # for scaling arrow lengths nicely
    mag = np.array(mag, dtype=float)
    mag[mag < 0] = 0.0
    if mapping == 'linear':
        mapped = mag
    elif mapping == 'sqrt':
        mapped = np.sqrt(mag)
    elif mapping == 'log':
        mapped = np.log10(mag + 1e-40)
        mapped = mapped - np.nanmin(mapped)
    else:
        raise ValueError("bad mapping type")
    maxv = np.nanmax(mapped)
    if maxv == 0:
        return np.zeros_like(mapped)
    return mapped / float(maxv)

def compute_on_radial_line(q, vx, r_array, theta):
    # gives E along a straight line at some angle (forward or sideways)
    if np.isclose(theta, 0.0):
        theta = 1e-8
    Rx = r_array * np.cos(theta)
    Ry = r_array * np.sin(theta)
    Ex, Ey = compute_E_field(q, vx, Rx, Ry)
    return np.sqrt(Ex**2 + Ey**2)

# -------------------------
# setup grids and speeds
# -------------------------
speeds = [0.1 * c, 0.9 * c, 0.999 * c]
labels = ["0.1c", "0.9c", "0.999c"]

x = np.linspace(-grid_halfwidth, grid_halfwidth, n_grid)
y = np.linspace(-grid_halfwidth, grid_halfwidth, n_grid)
X2, Y2 = np.meshgrid(x, y)

r_min, r_max, n_r = 0.01, grid_halfwidth, 800
r_array = np.linspace(r_min, r_max, n_r)

directions = {
    "Forward (θ≈0)": 0.0,
    "Perpendicular (θ=π/2)": np.pi / 2,
}

# -------------------------
# 2D plots for each velocity
# -------------------------
for vx, label in zip(speeds, labels):
    print(f"\n--- 2D plots for {label} ---")
    Ex2, Ey2 = compute_E_field(q, vx, X2, Y2)
    E_mag2 = np.sqrt(Ex2**2 + Ey2**2)
    Bx2, By2, Bz2 = compute_B_from_E(vx, Ex2, Ey2)
    Bmag2 = np.sqrt(Bx2**2 + By2**2 + Bz2**2)

    if not SHOW_PLOTS:
        continue

    # E field arrows
    plt.figure(figsize=(7,6))
    s = QUIVER_SUBSAMPLE
    plt.quiver(X2[::s,::s], Y2[::s,::s], Ex2[::s,::s], Ey2[::s,::s],
               color=ARROW_COLOR, width=QUIVER_WIDTH, pivot='mid', scale=None)
    plt.scatter(0,0,color='red',s=40,edgecolor='black',zorder=5,label='charge')
    plt.title(f"E-field vectors — v = {label}")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.axis('equal'); plt.legend(); plt.tight_layout(); plt.show()

    # |E| heatmap
    plt.figure(figsize=(6,5))
    plt.pcolormesh(X2, Y2, np.log10(E_mag2 + 1e-30), shading='auto', cmap='inferno')
    plt.colorbar(label='log10(|E|)')
    plt.scatter(0,0,color='white',s=30,edgecolor='black')
    plt.title(f"|E| heatmap — v = {label}")
    plt.axis('equal'); plt.tight_layout(); plt.show()

    # |E| vs distance
    plt.figure(figsize=(6,4))
    for dlabel, theta in directions.items():
        E_line = compute_on_radial_line(q, vx, r_array, theta)
        plt.loglog(r_array, E_line, label=dlabel)
    plt.xlabel("r (m)"); plt.ylabel("|E| (V/m)")
    plt.title(f"|E| vs distance — v = {label}")
    plt.grid(True,which='both',ls='--',alpha=0.6); plt.legend(); plt.tight_layout(); plt.show()

    # |B| heatmap
    plt.figure(figsize=(6,5))
    plt.pcolormesh(X2, Y2, np.log10(Bmag2 + 1e-40), shading='auto', cmap='plasma')
    plt.colorbar(label='log10(|B|)')
    plt.scatter(0,0,color='white',s=30,edgecolor='black')
    plt.title(f"|B| heatmap — v = {label}")
    plt.axis('equal'); plt.tight_layout(); plt.show()

# -------------------------
# 3D part (for all 3 speeds)
# -------------------------
x3 = np.linspace(-L3D, L3D, N3D)
y3 = np.linspace(-L3D, L3D, N3D)
z3 = np.linspace(-L3D, L3D, N3D)
X3, Y3, Z3 = np.meshgrid(x3, y3, z3, indexing='xy')

R3 = np.sqrt(X3**2 + Y3**2 + Z3**2)
mask3 = R3 > mask_cutoff

for vx, label in zip(speeds, labels):
    print(f"\n--- 3D plots for {label} ---")

    Ex3, Ey3, Ez3 = compute_E_field(q, vx, X3, Y3, Z3)
    Ex3_masked = np.where(mask3, Ex3, 0.0)
    Ey3_masked = np.where(mask3, Ey3, 0.0)
    Ez3_masked = np.where(mask3, Ez3, 0.0)

    Bx3, By3, Bz3 = compute_B_from_E(vx, Ex3_masked, Ey3_masked, Ez3_masked)

    E3_mag = np.sqrt(Ex3_masked**2 + Ey3_masked**2 + Ez3_masked**2)
    B3_mag = np.sqrt(Bx3**2 + By3**2 + Bz3**2)

    E_mapped = map_magnitude_to_unit(E3_mag, mapping=MAG_MAPPING)
    B_mapped = map_magnitude_to_unit(B3_mag, mapping=MAG_MAPPING)

    E3_mag_safe = np.where(E3_mag == 0, 1.0, E3_mag)
    B3_mag_safe = np.where(B3_mag == 0, 1.0, B3_mag)

    Ex_scaled = (Ex3_masked / E3_mag_safe) * (E_mapped * length3d)
    Ey_scaled = (Ey3_masked / E3_mag_safe) * (E_mapped * length3d)
    Ez_scaled = (Ez3_masked / E3_mag_safe) * (E_mapped * length3d)

    Bx_scaled = (Bx3 / B3_mag_safe) * (B_mapped * length3d)
    By_scaled = (By3 / B3_mag_safe) * (B_mapped * length3d)
    Bz_scaled = (Bz3 / B3_mag_safe) * (B_mapped * length3d)

    step3 = max(1, N3D // 8)
    Xs = X3[::step3, ::step3, ::step3]
    Ys = Y3[::step3, ::step3, ::step3]
    Zs = Z3[::step3, ::step3, ::step3]
    Exs = Ex_scaled[::step3, ::step3, ::step3]
    Eys = Ey_scaled[::step3, ::step3, ::step3]
    Ezs = Ez_scaled[::step3, ::step3, ::step3]
    Bxs = Bx_scaled[::step3, ::step3, ::step3]
    Bys = By_scaled[::step3, ::step3, ::step3]
    Bzs = Bz_scaled[::step3, ::step3, ::step3]
    masks_sub = mask3[::step3, ::step3, ::step3]

    X_e, Y_e, Z_e = Xs[masks_sub], Ys[masks_sub], Zs[masks_sub]
    U_e, V_e, W_e = Exs[masks_sub], Eys[masks_sub], Ezs[masks_sub]
    X_b, Y_b, Z_b = Xs[masks_sub], Ys[masks_sub], Zs[masks_sub]
    U_b, V_b, W_b = Bxs[masks_sub], Bys[masks_sub], Bzs[masks_sub]

    # E-field 3D plot
    fig = plt.figure(figsize=(8,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X_e, Y_e, Z_e, U_e, V_e, W_e, length=1.0, normalize=False, color='black')
    ax.set_xlim(-L3D, L3D); ax.set_ylim(-L3D, L3D); ax.set_zlim(-L3D, L3D)
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)'); ax.set_zlabel('z (m)')
    ax.set_title(f"3D E-field — v = {label} (arrow size ~ |E|)")
    plt.tight_layout(); plt.show()

    # B-field 3D plot
    figb = plt.figure(figsize=(8,7))
    axb = figb.add_subplot(111, projection='3d')
    axb.quiver(X_b, Y_b, Z_b, U_b, V_b, W_b, length=1.0, normalize=False, color='blue')
    axb.set_xlim(-L3D, L3D); axb.set_ylim(-L3D, L3D); axb.set_zlim(-L3D, L3D)
    axb.set_xlabel('x (m)'); axb.set_ylabel('y (m)'); axb.set_zlabel('z (m)')
    axb.set_title(f"3D B-field — v = {label} (arrow size ~ |B|)")
    plt.tight_layout(); plt.show()
