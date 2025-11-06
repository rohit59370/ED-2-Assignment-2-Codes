import numpy as np
import matplotlib.pyplot as plt

# Time range
t = np.linspace(0, 1000, 10000)

# Radiation reaction force (with constant set to 1)
F_rad = np.exp(t/100) * ((-900 + 1/10000) * np.sin(30*t) + (3/5) * np.cos(30*t))

# Plot
plt.figure(figsize=(10,5))
plt.plot(t, F_rad, linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Radiation Reaction Force (arbitrary units)')
plt.title('Radiation Reaction Force vs Time')
plt.grid(True)
plt.show()
