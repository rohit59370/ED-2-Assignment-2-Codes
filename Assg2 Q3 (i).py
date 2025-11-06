import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3D projection

# Parameters
dot_beta = 0.3
dot_beta_sq = dot_beta**2
beta_values = [0.1, 0.9, 0.999]
labels = ["beta = 0.1", "beta = 0.9", "beta = 0.999"]

# Theta: 0 to pi, Phi: 0 to 2pi
theta = np.linspace(0, np.pi, 200)
phi = np.linspace(0, 2*np.pi, 200)
TH, PH = np.meshgrid(theta, phi)

def angular_distribution(beta, TH, PH, dot_beta_sq):
    s = np.sin(TH)
    c = np.cos(TH)
    cp = np.cos(PH)
    sp = np.sin(PH)
    denom = (1 - beta * s * cp)
    # avoid division by very small numbers
    eps = 1e-9
    denom_safe = np.where(np.abs(denom) < eps, np.sign(denom)*eps, denom)
    numer = (1 - beta * s * cp)**2 * (c**2 + s**2 * sp**2) + (beta**2) * s**4 * cp**2
    return dot_beta_sq * numer / (denom_safe**5)

# Compute and plot separate 3D surface for each beta
for beta, label in zip(beta_values, labels):
    Z = angular_distribution(beta, TH, PH, dot_beta_sq)
    
    fig = plt.figure(figsize=(9,6))
    ax = fig.add_subplot(111, projection='3d')
    X = TH  # theta as x-axis
    Y = PH  # phi   as y-axis
    surf = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
    
    ax.set_title(f"Angular distribution (normalized) — {label}, dot_beta = {dot_beta}", fontsize=12)
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\phi$ (rad)")
    ax.set_zlabel(r"$\dfrac{dP}{d\Omega}$ (prefactor = 1)")
    ax.view_init(elev=25, azim=-60)
    plt.tight_layout()
    plt.show()
    # Optional: save the plot
    # plt.savefig(f"angular_beta_{str(beta).replace('.', '_')}.png", dpi=300)
