import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import Image

def plot_solution(t,x,u,xref=None,dt=0.02): #x and u in casadi (n_dof,n_timestep) format
    x = np.array(x)
    u = np.array(u)
    xref = np.array(xref)

    plt.figure(figsize=(12, 8))

    # 1) Angles vs ref
    plt.subplot(2, 2, 1)
    plt.plot(t, xref[:,2],'--', label='qmean_ref')
    plt.plot(t, x[:,2], label='qmean')
    plt.plot(t, xref[:,3],'--', label='qrel_ref')
    plt.plot(t, x[:,3], label='qrel')
    plt.axhline(y=0, color='grey')
    #plt.ylim(-2*np.pi, 2*np.pi)
    plt.xlabel('t [s]')
    plt.ylabel('angle [rad]')
    plt.title('Angles')
    plt.legend()

    # 2) Angular velocities vs ref
    plt.subplot(2, 2, 2)
    plt.plot(t, xref[:,6],'--', label='dqmean_ref')
    plt.plot(t, x[:,6], label='dqmean')
    plt.plot(t, xref[:,7],'--', label='dqrel_ref')
    plt.plot(t, x[:,7], label='dqrel')
    plt.axhline(y=0, color='grey')
    #plt.ylim(-10*np.pi, 10*np.pi)
    plt.xlabel('t [s]')
    plt.ylabel('angular vel [rad/s]')
    plt.title('Angular velocities')
    plt.legend()

    # 3) CM position
    plt.subplot(2, 2, 3)
    plt.plot(t, xref[:,0],'--', label='rCMy_ref')
    plt.plot(t, x[:,0], label='rCMy')
    plt.plot(t, xref[:,1],'--', label='rCMz_ref')
    plt.plot(t, x[:,1], label='rCMz')
    plt.axhline(y=0, color='grey')
    #plt.ylim(-2, 2)
    plt.xlabel('t [s]')
    plt.ylabel('position [m]')
    plt.title('Center of Mass')
    plt.legend()

    # 4) Control
    plt.subplot(2, 2, 4)
    plt.step(t[:-1], u[:,0], where='post')
    plt.axhline(y=0, color='grey')
    plt.xlabel('t [s]')
    #plt.ylim(-10, 10)
    plt.ylabel('u')
    plt.title('Control')

    plt.tight_layout()
    plt.show()

def show_pendulum(X, C1, C2, dt, p_val, gif_path="motion.gif"):
    """
    X   : (N, 8) array of states [rCMy,rCMz,qmean,qrel,vCMy,vCMz,dqmean,dqrel]
    C1  : (N,) contact flag 1
    C2  : (N,) contact flag 2
    ts  : (N,) time vector (only used for consistency / potential future use)
    dt  : scalar time step (used for playback speed)
    p : DM or array with parameters [m,M,l,k,kappa,J,g]
    """

    X = np.asarray(X)
    C1 = np.asarray(C1).flatten()
    C2 = np.asarray(C2).flatten()

    N = X.shape[0]           # number of time steps

    # --- Forward kinematics (same as before) ---
    rCMy, rCMz, qmean, qrel = ca.MX.sym('rCMy'), ca.MX.sym('rCMz'), ca.MX.sym('qmean'), ca.MX.sym('qrel')
    q = ca.vertcat(rCMy, rCMz, qmean, qrel)

    m, M, l, k, kappa, Jpar, g = ca.MX.sym('m'), ca.MX.sym('M'), ca.MX.sym('l'), ca.MX.sym('k'), ca.MX.sym('kappa'), ca.MX.sym('Jpar'), ca.MX.sym('g')
    p = ca.vertcat(m, M, l, k, kappa, Jpar, g)

    phi1 = qmean - 0.5*qrel
    phi2 = qmean + 0.5*qrel

    rBy = rCMy + 2*m*l/(M+2*m)*ca.sin(qmean)*ca.cos(qrel/2)
    rBz = rCMz + 2*m*l/(M+2*m)*ca.cos(qmean)*ca.cos(qrel/2)
    r1y = rBy - l*ca.sin(phi1)
    r1z = rBz - l*ca.cos(phi1)
    r2y = rBy - l*ca.sin(phi2)
    r2z = rBz - l*ca.cos(phi2)

    fk_fun = ca.Function('fk_fun', [q, p], [rBy, rBz, r1y, r1z, r2y, r2z])

    # --- Evaluate forward kinematics for each time step ---
    YB, ZB, Y1, Z1, Y2, Z2 = [], [], [], [], [], []
    for k in range(N):
        qk = ca.DM(X[k, [0, 1, 2, 3]])            # [rCMy,rCMz,qmean,qrel]
        yb, zb, y1, z1, y2, z2 = fk_fun(qk, p_val)
        YB.append(float(yb)); ZB.append(float(zb))
        Y1.append(float(y1)); Z1.append(float(z1))
        Y2.append(float(y2)); Z2.append(float(z2))

    YB, ZB, Y1, Z1, Y2, Z2 = map(np.array, (YB, ZB, Y1, Z1, Y2, Z2))

    # --- Setup figure and artists ---
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xlim(-0.5, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axhline(0, color='grey', lw=1)

    leg1, = ax.plot([], [], 'o-', lw=2, color='tab:blue')
    leg2, = ax.plot([], [], 'o-', lw=2, color='tab:orange')
    cm_pt, = ax.plot([], [], 'o', ms=6, label='CM',color='tab:green')
    contact_dot1, = ax.plot([], [], 'ro', ms=5)
    contact_dot2, = ax.plot([], [], 'ro', ms=5)
    ax.legend()

    def init():
        leg1.set_data([], []); leg2.set_data([], [])
        cm_pt.set_data([], []); contact_dot1.set_data([], []); contact_dot2.set_data([], [])
        return leg1, leg2, cm_pt, contact_dot1, contact_dot2

    def update(i):
        leg1.set_data([YB[i], Y1[i]], [ZB[i], Z1[i]])
        leg2.set_data([YB[i], Y2[i]], [ZB[i], Z2[i]])
        cm_pt.set_data([float(X[i,0])], [float(X[i,1])])

        if C1[i] > 0.5:
            contact_dot1.set_data([float(Y1[i])], [0.0])
        else:
            contact_dot1.set_data([], [])
        if C2[i] > 0.5:
            contact_dot2.set_data([float(Y2[i])], [0.0])
        else:
            contact_dot2.set_data([], [])
        return leg1, leg2, cm_pt, contact_dot1, contact_dot2

    ani = FuncAnimation(fig, update, frames=N,
                        init_func=init, blit=True,
                        interval=dt * 1000, repeat=False)

    ani.save(gif_path, writer='pillow', fps=int(1/dt))
    plt.close(fig)
    return Image(filename=gif_path)


def plot_eigs_unit_circle(eigvals, ax=None, title="Discrete-time eigenvalues"):
    """
    Plot eigenvalues in the complex plane with the unit circle.

    Parameters
    ----------
    eigvals : array-like
        Iterable of complex eigenvalues.
    ax : matplotlib axis, optional
        If provided, plot into this axis.
    title : str
        Plot title.
    """
    eigvals = np.asarray(eigvals)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    # Unit circle
    theta = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), 'r--', lw=1, label='Unit circle')

    # Eigenvalues
    ax.scatter(eigvals.real, eigvals.imag, c='green', s=60, zorder=3, label='Eigenvalues')

    # Axes formatting
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(0, color='k', lw=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imag')
    ax.set_title(title)

    # Limits (auto, but symmetric)
    r = max(1.1, np.max(np.abs(eigvals))*1.1)
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])

    ax.legend()
    return ax
