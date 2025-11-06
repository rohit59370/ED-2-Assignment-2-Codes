import numpy as np
import matplotlib.pyplot as plt

# Theta range: 0 to pi
theta = np.linspace(0, np.pi, 1000)

# Constants
dot_beta = 0.9
factor = dot_beta**2  # (a/c)^2 = 0.9^2 = 0.81

# Angular power distribution function
def dP_dOmega(beta, theta):
    return factor * (np.sin(theta))**2 / (1 - beta * np.cos(theta))**5

# Different velocities
beta_values = [0.1, 0.9, 0.999]
labels = ["v = 0.1c", "v = 0.9c", "v = 0.999c"]

# Generate individual plots
for beta, label in zip(beta_values, labels):
    plt.figure(figsize=(7, 5))
    plt.plot(theta, dP_dOmega(beta, theta), lw=2)
    plt.title(f"Angular Distribution for {label}", fontsize=14)
    plt.xlabel(r"$\theta$ (radians)", fontsize=12)
    plt.ylabel(r"$\frac{dP}{d\Omega}$ (normalized)", fontsize=12)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()
