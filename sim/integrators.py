from dataclasses import dataclass
import numpy as np

Array = np.ndarray

def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_normalize(q):
    n = np.linalg.norm(q);
    return q if n == 0 else q/n

def _quat_from_omega_dt(omega, dt):
    th = np.linalg.norm(omega) * dt
    if th < 1e-12:
        return _quat_normalize(np.array([1.0, 0.5*dt*omega[0], 0.5*dt*omega[1], 0.5*dt*omega[2]]))
    ax = omega / (np.linalg.norm(omega) + 1e-16)
    s = np.sin(0.5*th)
    return np.array([np.cos(0.5*th), ax[0]*s, ax[1]*s, ax[2]*s])

def _R_from_quat(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1-2*(x*x+z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),   2*(y*z + w*x), 1-2*(x*x+y*y)],
    ])

class Integrator:
    def step(self, f, t, x, dt): raise NotImplementedError

@dataclass
class LGVI(Integrator):
    m: float
    J: Array
    newton_iters: int = 2
    # NEW: tell LGVI how to split [qpos, qvel]
    nq: int | None = None
    nv: int | None = None

    def step(self, wrench_fn, t: float, x: Array, dt: float):
        assert self.nq is not None and self.nv is not None, "LGVI needs nq and nv"
        nq, nv = self.nq, self.nv

        # ---- split [qpos, qvel] ----
        qpos = x[:nq].copy()
        qvel = x[nq:nq+nv].copy()

        # free-joint components (first 7 in qpos, first 6 in qvel)
        p = qpos[0:3]
        q = _quat_normalize(qpos[3:7])
        v = qvel[0:3]
        w = qvel[3:6]

        R = _R_from_quat(q)

        # forces/torques at k
        Wk = wrench_fn(t, x)                     # expects full state
        Fk_w = Wk["F_world"]
        tauk_b = (R.T @ Wk["tau_world"]) if "tau_world" in Wk else Wk["tau_body"]

        # ---- linear half-step ----
        v_half = v + (dt/(2.0*self.m)) * Fk_w

        # ---- rotational half-step (fixed-point) ----
        omega_half = w.copy()
        for _ in range(self.newton_iters):
            cori = np.cross(omega_half, self.J @ omega_half)
            rhs  = tauk_b - cori
            omega_half = w + (dt/2.0) * np.linalg.solve(self.J, rhs)

        # ---- update pose with half-step velocities ----
        dq = _quat_from_omega_dt(omega_half, dt)
        q_next = _quat_normalize(_quat_mul(q, dq))
        p_next = p + dt * v_half

        # ---- build x_prov as a FULL state to query forces at k+1 ----
        x_prov = x.copy()
        qpos_prov = qpos.copy(); qpos_prov[0:3] = p_next; qpos_prov[3:7] = q_next
        qvel_prov = qvel.copy(); qvel_prov[0:3] = v_half; qvel_prov[3:6] = omega_half
        x_prov[:nq] = qpos_prov
        x_prov[nq:nq+nv] = qvel_prov

        Wn = wrench_fn(t + dt, x_prov)
        Fn_w = Wn["F_world"]
        # need R at k+1 for world→body torque
        Rn = _R_from_quat(q_next)
        taun_b = (Rn.T @ Wn["tau_world"]) if "tau_world" in Wn else Wn["tau_body"]

        # complete updates
        v_next = v_half + (dt/(2.0*self.m)) * Fn_w
        cori_half = np.cross(omega_half, self.J @ omega_half)
        rhs_n = taun_b - cori_half
        w_next = w + dt * np.linalg.solve(self.J, rhs_n)

        # ---- pack x_next as FULL state (pass-through all other DoFs) ----
        x_next = x.copy()
        qpos_next = qpos.copy(); qpos_next[0:3] = p_next; qpos_next[3:7] = q_next
        qvel_next = qvel.copy(); qvel_next[0:3] = v_next; qvel_next[3:6] = w_next
        x_next[:nq] = qpos_next
        x_next[nq:nq+nv] = qvel_next

        return t + dt, x_next