import numpy as np
import casadi as ca
from scipy.linalg import solve_discrete_are,eig
from integrate import make_stepper,linearize_discrete
from dynamics_hybrid import f,rphi_to_q,drphi_to_dq

def LQR(x_d,u_d,Q,R,p,c,dt):
    # get Jacobians as an expression
    F = make_stepper(f, dt)
    A_func, B_func = linearize_discrete(F)
    A = A_func(x_d, u_d, p, c[0], c[1])
    B = B_func(x_d, u_d, p, c[0], c[1])

    # take controllable part of state: [qmean,qrel,dqmean,dqrel]
    idx = [2,3,6,7]
    A_ctrl = np.array(A)[np.ix_(idx, idx)]
    B_ctrl = np.array(B)[np.ix_(idx, [0])]
    print('Shape of controllable A:', A.shape)
    print('Shape of controllable B:', B.shape)

    P = solve_discrete_are(A_ctrl, B_ctrl, Q, R)
    K = np.linalg.solve(R + B_ctrl.T @ P @ B_ctrl, B_ctrl.T @ P @ A_ctrl)

    K = ca.DM(K)  # convert back to casadi format
    #A_ctrl = ca.DM(A_ctrl)
    #B_ctrl = ca.DM(B_ctrl)
    print('Shape of feedback gain K:', K.shape)

    return K,A_ctrl,B_ctrl,A,B

def initial_from_eig(amp,rtip,c,p,A_ctrl,dt):
    evals, evecs = eig(A_ctrl)
    id_eig = np.argmax(np.abs(np.imag(evals))) #eval with largest positive imag part
    freq = np.angle(evals[id_eig]) / (2 * np.pi * dt) #calculate corresponding freq
    #T_period = 1 / abs(freq)
    #N_period = T_period / dt
    evec_norm = np.real(evecs[:, id_eig] / np.linalg.norm(evecs[:, id_eig]))

    #construct initial state
    qmean, qrel, dqmean, dqrel = amp * evec_norm
    phi1 = qmean - qrel / 2
    phi2 = qmean + qrel / 2
    phi = np.array([phi1, phi2])
    dphi1 = dqmean - dqrel / 2
    dphi2 = dqmean + dqrel / 2
    dphi = np.array([dphi1, dphi2])
    q_init = rphi_to_q(c, rtip, phi, p)
    dq_init = drphi_to_dq(c, dphi, q_init, p)
    x_init = ca.vertcat(q_init, dq_init)

    return x_init,freq

def forward_linear(x_init, p, c, dt, N, u_init=0, x_d=None, u_d=None, Q=None, R=None, LQR_ctrl=False):
    K,A_ctrl,B_ctrl,A,B = LQR(x_d,u_d,Q,R,p,c,dt)
    idx = [2,3,6,7]
    dx = ca.DM.zeros(8, N + 1)
    du = ca.DM.zeros(1, N)
    dx[:, 0] = x_init - x_d
    du[:, 0] = u_init - u_d
    for k in range(N):  # feedback control loop
        if LQR_ctrl:
            du[:, k] = - K @ dx[idx, k]  # controls are chosen only from the controllable state part
        else:
            du[:, k] = 0
        dx[:, k + 1] = ca.mtimes(A, dx[:, k]) + ca.mtimes(B, du[:, k])

    # reconstructed linear state
    x = np.array(ca.repmat(x_d, 1, N + 1) + dx).T
    u = np.array(ca.repmat(u_d, 1, N) + du).T
    t = np.linspace(0, N * dt, N + 1)

    return t, x, u