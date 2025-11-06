import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def pos_time(beta, beta_dot, c=3e8):
    # figuring out how long it takes till particle is ~1414 m away (our cutoff)
    def r_mag(t):
        pos = c * (beta * t + 0.5 * beta_dot * t**2)
        return np.linalg.norm(pos)

    def f(t):
        return r_mag(t) - 1414.0

    # just a rough guess for time limit so root solver behaves
    t_max = (1 - np.linalg.norm(beta)) / (np.linalg.norm(beta_dot) + 1e-12)

    # finding when |r| = 1414 using brent root solver
    sol = root_scalar(f, bracket=[0, t_max], method='brentq')
    T = sol.root

    # generate time array
    t = np.linspace(0, T, 1000)

    # now position arrays
    x = c * (beta[0]*t + 0.5*beta_dot[0]*t**2)
    y = c * (beta[1]*t + 0.5*beta_dot[1]*t**2)
    z = c * (beta[2]*t + 0.5*beta_dot[2]*t**2)
    return t, x, y, z


def fields(beta, beta_dot, obs, c=3e8, q=1, factor=9e9):
    # get time and position info
    t, x, y, z = pos_time(beta, beta_dot, c)

    E = np.zeros((len(t), 3))
    B = np.zeros((len(t), 3))

    for i in range(len(t)):
        # vector from particle to observer
        r_vec = obs - np.array([x[i], y[i], z[i]])
        R = np.linalg.norm(r_vec)
        n = r_vec / R

        # instantaneous velocity
        b = beta + beta_dot * t[i]

        # main radiation field equation
        num = np.cross(n, np.cross((n - b), beta_dot))
        denom = (1 - np.dot(n, b))**3 * R
        E[i] = (q * factor / c) * num / denom
        B[i] = np.cross(n, E[i]) / c

    return t, E[:, 0], E[:, 1], E[:, 2], B[:, 0], B[:, 1], B[:, 2], x, y, z


def group_similar_components(components, threshold=0.4, eps=1e-20):
    # this groups together components that vary roughly in the same range
    ranges = [max(np.max(a) - np.min(a), eps) for a in components]
    unassigned = set(range(len(components)))
    groups = []
    while unassigned:
        i = unassigned.pop()
        group = [i]
        for j in list(unassigned):
            diff = abs(ranges[i] - ranges[j]) / max(ranges[i], ranges[j])
            if diff < threshold:
                group.append(j)
                unassigned.remove(j)
        groups.append(sorted(group))
    groups.sort(key=lambda g: g[0])
    return groups


def plot_fields_adaptive(beta, beta_dot, obs, similarity_threshold=0.4):
    # compute fields and positions
    t, Ex, Ey, Ez, Bx, By, Bz, x, y, z = fields(beta, beta_dot, obs)

    # nice clean string format for titles
    fmt = lambda arr: ", ".join(f"{float(v):.3f}" for v in arr)
    beta_str = fmt(beta)
    beta_dot_str = fmt(beta_dot)

    # field components
    E_list = [Ex, Ey, Ez]
    E_labels = ['Ex', 'Ey', 'Ez']
    E_colors = ['tab:red', 'tab:green', 'tab:blue']
    E_styles = ['-', '--', '-.']

    B_list = [Bx, By, Bz]
    B_labels = ['Bx', 'By', 'Bz']
    B_colors = ['tab:purple', 'tab:cyan', 'orange']
    B_styles = ['-', '--', '-.']

    # figure out which components go together based on range
    E_groups = group_similar_components(E_list, threshold=similarity_threshold)
    B_groups = group_similar_components(B_list, threshold=similarity_threshold)

    # --------- plot E fields ---------
    figE, axesE = plt.subplots(len(E_groups), 1, figsize=(8, 3 * len(E_groups)), sharex=True)
    if len(E_groups) == 1:
        axesE = [axesE]
    for ax, grp in zip(axesE, E_groups):
        for idx in grp:
            ax.plot(t, E_list[idx], label=E_labels[idx],
                    color=E_colors[idx], linestyle=E_styles[idx], linewidth=1.6)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='small')
        ax.set_ylabel("E (N/C)")
        ax.set_title(" & ".join([E_labels[i] for i in grp]) + " vs Time")
    axesE[-1].set_xlabel("Time (s)")
    figE.suptitle(f"Electric Fields — β={beta_str}, β̇={beta_dot_str}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --------- plot B fields ---------
    figB, axesB = plt.subplots(len(B_groups), 1, figsize=(8, 3 * len(B_groups)), sharex=True)
    if len(B_groups) == 1:
        axesB = [axesB]
    for ax, grp in zip(axesB, B_groups):
        for idx in grp:
            ax.plot(t, B_list[idx], label=B_labels[idx],
                    color=B_colors[idx], linestyle=B_styles[idx], linewidth=1.6)
        ax.grid(True)
        ax.legend(loc='upper right', fontsize='small')
        ax.set_ylabel("B (T)")
        ax.set_title(" & ".join([B_labels[i] for i in grp]) + " vs Time")
    axesB[-1].set_xlabel("Time (s)")
    figB.suptitle(f"Magnetic Fields — β={beta_str}, β̇={beta_dot_str}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # quick print check for sanity
    print("Check: |E| ≈ c|B| at t[10]")
    i = 10
    print("E_mag =", np.linalg.norm([Ex[i], Ey[i], Ez[i]]))
    print("c*B_mag =", 3e8 * np.linalg.norm([Bx[i], By[i], Bz[i]]))

    # --------- 3D trajectory plot (clean white look) ---------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')

    # blue dashed line for trajectory
    ax.plot(x, y, z, '--', color='blue', linewidth=2.2, label='Trajectory')

    # start, end and observer points
    ax.scatter(x[0], y[0], z[0], color='lime', s=70, edgecolors='k', linewidths=0.8, label='Start')
    ax.scatter(x[-1], y[-1], z[-1], color='red', s=70, edgecolors='k', linewidths=0.8, label='End')
    ax.scatter(obs[0], obs[1], obs[2], color='purple', s=140, marker='s',
               edgecolors='k', linewidths=0.9, label='Observer')

    # white background, normal grid on
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    ax.grid(True)

    # better viewing angle
    ax.view_init(elev=22, azim=135)

    # equal-ish axis scales
    try:
        max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        midx = 0.5 * (x.max() + x.min())
        midy = 0.5 * (y.max() + y.min())
        midz = 0.5 * (z.max() + z.min())
        ax.set_xlim(midx - max_range/2, midx + max_range/2)
        ax.set_ylim(midy - max_range/2, midy + max_range/2)
        ax.set_zlim(midz - max_range/2, midz + max_range/2)
    except Exception:
        pass

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_title(f'3D Trajectory — β={beta_str}, β̇={beta_dot_str}')
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# --------- running both parts ---------
if __name__ == "__main__":
    # Part A: accel along z
    for v in [0.1, 0.9, 0.999]:
        beta = np.array([v, 0, 0])
        beta_dot = np.array([0, 0, 0.3])
        obs = np.array([1000, 1000, 0])
        plot_fields_adaptive(beta, beta_dot, obs)

    # Part B: accel along x
    for v in [0.1, 0.9, 0.999]:
        beta = np.array([v, 0, 0])
        beta_dot = np.array([0.9, 0, 0])
        obs = np.array([1000, 1000, 0])
        plot_fields_adaptive(beta, beta_dot, obs)
