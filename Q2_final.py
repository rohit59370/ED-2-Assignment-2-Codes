import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ------------------ trajectory / timing ------------------
def pos_time(beta: np.ndarray, beta_dot: np.ndarray, c=3e8):
    """Return time array and particle coordinates (x,y,z) for motion
    r(t) = c * (beta*t + 0.5*beta_dot*t^2) with integration stopping
    when ||r|| reaches 1414.0 (keeps original behaviour)."""
    def radius(t):
        pos = c * (beta * t + 0.5 * beta_dot * t**2)
        return np.linalg.norm(pos)

    def root_fun(t):
        return radius(t) - 1414.0

    t_max = (1.0 - np.linalg.norm(beta)) / (np.linalg.norm(beta_dot) + 1e-12)
    sol = root_scalar(root_fun, bracket=[0.0, t_max], method='brentq')
    T = sol.root

    times = np.linspace(0.0, T, 1000)
    pos = c * (np.outer(times, beta) + 0.5 * np.outer(times**2, beta_dot))
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    return times, x, y, z


# ------------------ fields (Liénard–Wiechert radiation term) ------------------
def fields(beta: np.ndarray, beta_dot: np.ndarray, obs: np.ndarray,
           c=3e8, q=1.0, factor=9e9):
    """Compute E(t) and B(t) fields at observer for the moving charge.
    Returns time array, components Ex,Ey,Ez,Bx,By,Bz and trajectory coords x,y,z."""
    t, x, y, z = pos_time(beta, beta_dot, c)
    nT = len(t)
    E = np.zeros((nT, 3))
    B = np.zeros((nT, 3))

    for i in range(nT):
        src = np.array([x[i], y[i], z[i]])
        r_vec = obs - src
        R = np.linalg.norm(r_vec)
        n_hat = r_vec / R

        beta_inst = beta + beta_dot * t[i]
        numerator = np.cross(n_hat, np.cross((n_hat - beta_inst), beta_dot))
        denominator = (1.0 - np.dot(n_hat, beta_inst))**3 * R

        E[i, :] = (q * factor / c) * numerator / denominator
        B[i, :] = np.cross(n_hat, E[i, :]) / c

    Ex, Ey, Ez = E[:, 0], E[:, 1], E[:, 2]
    Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
    return t, Ex, Ey, Ez, Bx, By, Bz, x, y, z


# ------------------ grouping utility ------------------
def group_similar_components(components, threshold=0.4, eps=1e-20):
    """
    Greedy grouping of components with similar ranges.
    components : list of 1D arrays
    threshold : relative-difference threshold to consider two ranges 'similar'
    Returns list of groups (each group is a sorted list of indices).
    """
    ranges = []
    for arr in components:
        r = np.max(arr) - np.min(arr)
        ranges.append(max(r, eps))
    unassigned = set(range(len(components)))
    groups = []
    while unassigned:
        i = unassigned.pop()
        group = [i]
        to_check = list(unassigned)
        for j in to_check:
            rel_diff = abs(ranges[i] - ranges[j]) / max(ranges[i], ranges[j])
            if rel_diff < threshold:
                group.append(j)
                unassigned.remove(j)
        groups.append(sorted(group))
    groups.sort(key=lambda g: g[0])
    return groups


# ------------------ plotting with adaptive grouping & fixed titles ------------------
def plot_fields_adaptive(beta: np.ndarray, beta_dot: np.ndarray, obs: np.ndarray,
                         similarity_threshold=0.4):
    """Compute fields and plot them using adaptive grouping; fixes np.float titles."""
    # compute
    t, Ex, Ey, Ez, Bx, By, Bz, x, y, z = fields(beta, beta_dot, obs)

    # pretty format for beta / beta_dot to avoid np.float64 reprs in titles
    def fmt_vec(arr, fmt="{:.6g}"):
        return ", ".join(fmt.format(float(v)) for v in np.asarray(arr).ravel())

    beta_str = fmt_vec(beta)
    beta_dot_str = fmt_vec(beta_dot)

    # prepare lists
    E_list = [Ex, Ey, Ez]
    E_labels = ['Ex', 'Ey', 'Ez']
    E_colors = ['tab:red', 'tab:green', 'tab:blue']
    E_styles = ['-', '--', '-.']

    B_list = [Bx, By, Bz]
    B_labels = ['Bx', 'By', 'Bz']
    B_colors = ['tab:purple', 'tab:cyan', 'orange']
    B_styles = ['-', '--', '-.']

    # grouping
    E_groups = group_similar_components(E_list, threshold=similarity_threshold)
    B_groups = group_similar_components(B_list, threshold=similarity_threshold)

    # ---- plot E fields: one figure, multiple vertically stacked axes depending on grouping ----
    nE_subplots = len(E_groups)
    figE, axesE = plt.subplots(nE_subplots, 1, figsize=(8, 3 * nE_subplots), sharex=True)
    if nE_subplots == 1:
        axesE = [axesE]
    for ax, grp in zip(axesE, E_groups):
        for idx in grp:
            ax.plot(t, E_list[idx], label=E_labels[idx],
                    color=E_colors[idx], linestyle=E_styles[idx], linewidth=1.6)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='small')
        if len(grp) == 1:
            ax.set_ylabel(f"{E_labels[grp[0]]} (N/C)")
            ax.set_title(f"{E_labels[grp[0]]} vs Time")
        else:
            ax.set_ylabel("E (N/C)")
            ax.set_title(" & ".join([E_labels[i] for i in grp]) + " vs Time")
    axesE[-1].set_xlabel("Time (s)")
    figE.suptitle(f"Electric field components — β={beta_str}, β̇={beta_dot_str}", fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ---- plot B fields similarly ----
    nB_subplots = len(B_groups)
    figB, axesB = plt.subplots(nB_subplots, 1, figsize=(8, 3 * nB_subplots), sharex=True)
    if nB_subplots == 1:
        axesB = [axesB]
    for ax, grp in zip(axesB, B_groups):
        for idx in grp:
            ax.plot(t, B_list[idx], label=B_labels[idx],
                    color=B_colors[idx], linestyle=B_styles[idx], linewidth=1.6)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='small')
        if len(grp) == 1:
            ax.set_ylabel(f"{B_labels[grp[0]]} (T)")
            ax.set_title(f"{B_labels[grp[0]]} vs Time")
        else:
            ax.set_ylabel("B (T)")
            ax.set_title(" & ".join([B_labels[i] for i in grp]) + " vs Time")
    axesB[-1].set_xlabel("Time (s)")
    figB.suptitle(f"Magnetic field components — β={beta_str}, β̇={beta_dot_str}", fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # ---- sanity check ----
    print("A quick test: |E| ≈ c|B| for radiation field. Checking at t[10]:")
    i = 10
    print("E_mag =", np.linalg.norm([Ex[i], Ey[i], Ez[i]]))
    print("c*B_mag =", 3e8 * np.linalg.norm([Bx[i], By[i], Bz[i]]))

    # ---- 3D trajectory ----
    fig = plt.figure(figsize=(6, 5))
    ax3 = fig.add_subplot(111, projection='3d')
    ax3.plot(x, y, z, '--', color='blue', linewidth=1.8, label='Particle trajectory')  # dashed blue
    ax3.scatter(x[0], y[0], z[0], color='green', s=12, label='Start')
    ax3.scatter(x[-1], y[-1], z[-1], color='red', s=12, label='End')
    ax3.scatter(obs[0], obs[1], obs[2], color='purple', s=80, marker='s', label='Observer')  # purple square
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    ax3.set_zlabel('z (m)')
    ax3.set_title(f'Particle trajectory and observer position — β={beta_str}, β̇={beta_dot_str}')
    ax3.legend()
    ax3.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------ example runs (same parameter sets as original) ------------------
if __name__ == "__main__":
    # PART A
    for v in [0.1, 0.9, 0.999]:
        beta = np.array([v, 0.0, 0.0])
        beta_dot = np.array([0.0, 0.0, 0.3])
        obs = np.array([1000.0, 1000.0, 0.0])
        plot_fields_adaptive(beta, beta_dot, obs, similarity_threshold=0.4)

    # PART B
    for v in [0.1, 0.9, 0.999]:
        beta = np.array([v, 0.0, 0.0])
        beta_dot = np.array([0.9, 0.0, 0.0])
        obs = np.array([1000.0, 1000.0, 0.0])
        plot_fields_adaptive(beta, beta_dot, obs, similarity_threshold=0.4)
