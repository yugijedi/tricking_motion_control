import casadi as ca
import numpy as np

def make_stepper(f, dt):
    """
    Returns a CasADi one-step integrator Function:
      xnext = F(x,u,p,c1,c2)
    using dxdt from f(x,u,p,c1,c2).
    """
    f = f.expand()
    x, u, p, c1, c2 = f.mx_in()
    dxdt, lam_out, c1_next, c2_next, dq_plus = f(x, u, p, c1, c2)

    ode = {'x': x, 'u': u, 'p': ca.vertcat(p, c1, c2), 'ode': dxdt}
    intg = ca.integrator('intg', 'rk', ode, 0, dt, {'simplify': True})
    res = intg(x0=x, u=u, p=ca.vertcat(p, c1, c2))

    F = ca.Function('F_step',
                    [x, u, p, c1, c2],
                    [res['xf']],
                    ['x','u','p','c1','c2'],
                    ['xnext'])
    return F


def linearize_discrete(F):
    x, u, p, c1, c2 = F.mx_in()
    xnext = F(x, u, p, c1, c2)
    # ensure it's the actual expression (some CasADi versions return a list/tuple)
    if isinstance(xnext, (list, tuple)):
        xnext = xnext[0]
    Ad = ca.jacobian(xnext, x)
    Bd = ca.jacobian(xnext, u)
    Ad_fun = ca.Function('Ad_fun', [x, u, p, c1, c2], [Ad],
                         ['x', 'u', 'p', 'c1', 'c2'], ['Ad'])
    Bd_fun = ca.Function('Bd_fun', [x, u, p, c1, c2], [Bd],
                         ['x', 'u', 'p', 'c1', 'c2'], ['Bd'])
    return Ad_fun, Bd_fun


def forward(x0, u, p, dt, N, c0=(0.0, 0.0), f_dyn=None, stop_condition=None):
    """
    Forward simulation with optional hybrid dynamics and early stopping.

    Parameters
    ----------
    x0 : casadi.DM (8x1)
        Initial state.
    u  : array-like, shape (N, 1) or (1, N)
        Control sequence over N steps.
    p  : casadi.DM or array-like
        Parameter vector.
    dt : float
        Time step.
    N  : int
        Maximum number of steps.
    c0 : tuple(float,float)
        Initial contact flags (c1,c2).
    f_dyn : casadi.Function
        Hybrid dynamics function f(x,u,p,c1,c2).
    stop_condition : callable or None
        Optional function:
            stop_condition(x_prev, x_curr, c_curr, t_curr, k_curr) -> bool
        If it returns True, integration stops at step k_curr.

    Returns
    -------
    X_hist : (K+1, 8) ndarray
        State trajectory up to stopping index K (<= N).
    t_hist : (K+1,) ndarray
        Time stamps.
    C_hist : (K+1, 2) ndarray
        Contact flags.
    k_td   : int or None
        First touchdown step index (k+1) or None.
    k_stop : int or None
        Index where stop_condition triggered (if any), else None.
    """
    if f_dyn is None:
        raise ValueError("Pass your dynamics function as f_dyn=f.")

    F = make_stepper(f_dyn, dt)

    c1, c2 = float(c0[0]), float(c0[1])
    x = x0

    X_hist = np.zeros((N+1, 8), dtype=float)
    C_hist = np.zeros((N+1, 2), dtype=float)
    t_hist = np.zeros((N+1,), dtype=float)
    u = np.array(u.T)  # assume u is (1,N) or (N,1); make it (N,)

    X_hist[0, :] = np.array(x.T).flatten()
    C_hist[0, :] = [c1, c2]
    t_hist[0] = 0.0

    k_td = None
    k_stop = None

    for k in range(N):
        # predict contact transition and (if touchdown) dq_plus
        _, _, c1_next, c2_next, dq_plus = f_dyn(x, u[k], p, c1, c2)
        c1n = float(c1_next)
        c2n = float(c2_next)

        touchdown = ((c1 < 0.5 and c1n > 0.5) or (c2 < 0.5 and c2n > 0.5))
        if touchdown:
            x = ca.vertcat(x[0:4], dq_plus)  # apply impact velocity jump
            if k_td is None:
                k_td = k + 1

        # integrate with current mode
        x_prev = X_hist[k, :].copy()
        x = F(x, u[k], p, c1, c2)

        # update mode for next step
        c1, c2 = c1n, c2n

        X_hist[k+1, :] = np.array(x.T).flatten()
        C_hist[k+1, :] = [c1, c2]
        t_hist[k+1] = (k+1) * dt

        # early stopping: Poincaré / symmetry section etc.
        if stop_condition is not None:
            if stop_condition(x_prev, X_hist[k+1, :], C_hist[k+1, :], t_hist[k+1], k+1):
                k_stop = k + 1
                # truncate histories to [0..k_stop]
                X_hist = X_hist[:k_stop+1, :]
                C_hist = C_hist[:k_stop+1, :]
                t_hist = t_hist[:k_stop+1]
                break

    # if no early stop, k_stop stays None; histories already full length
    return X_hist, t_hist, C_hist, k_td, k_stop