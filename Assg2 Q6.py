import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, k, pi

# Spectral shape function S(nu; T) up to an overall constant
def S_nu(nu, T):
    x = (h * nu) / (k * T)
    # avoid tiny numerical issues at extremely small x
    x = np.maximum(x, 1e-30)
    shape = np.exp(-x) * (2.2 + 2.2*x + 0.6*x**2)
    # include the T^{-1/2} factor (from the derivation)
    return shape * T**(-0.5)

# temperatures
T1 = 300.0          # K
T2 = 1e6            # K

# global frequency array (Hz)
nu = np.logspace(9, 18, 1000)   # 1e9 ... 1e18 Hz

S1 = S_nu(nu, T1)
S2 = S_nu(nu, T2)

# normalized-to-peak curves (for shape comparison)
S1_norm = S1 / S1.max()
S2_norm = S2 / S2.max()

# --- Plot 1: raw (includes T^(-1/2)) ---
plt.figure(figsize=(8,5))
plt.loglog(nu, S1, label=f'T = {T1:g} K (raw)')
plt.loglog(nu, S2, label=f'T = {T2:.0e} K (raw)')
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel(r'Spectral power (arb. units)')
plt.title('Bremsstrahlung spectral power (raw, same normalization constants)')
plt.legend()
plt.grid(which='both', ls=':', alpha=0.4)
plt.tight_layout()
plt.show()

# --- Plot 2: normalized to peak (compare shapes / peak positions) ---
plt.figure(figsize=(8,5))
plt.semilogx(nu, S1_norm, label=f'T = {T1:g} K (normalized)')
plt.semilogx(nu, S2_norm, label=f'T = {T2:.0e} K (normalized)')
plt.xlabel(r'Frequency $\nu$ (Hz)')
plt.ylabel('Normalized spectral power (peak = 1)')
plt.title('Bremsstrahlung spectral shape (normalized)')
plt.legend()
plt.grid(which='both', ls=':', alpha=0.4)
plt.tight_layout()
plt.show()
