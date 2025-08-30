import numpy as np

Array = np.ndarray

def _quat_mul(q1: Array, q2: Array) -> Array:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_normalize(q: Array) -> Array:
    return q / np.linalg.norm(q)

def _quat_from_omega_dt(omega: Array, dt: float) -> Array:
    # omega is body angular velocity (rad/s)
    theta = np.linalg.norm(omega) * dt
    if theta < 1e-12:
        # first-order: cos ~ 1, sin(theta/2)/(theta) ~ 0.5
        half = 0.5 * dt
        return _quat_normalize(np.array([1.0, half*omega[0], half*omega[1], half*omega[2]]))
    axis = omega / (np.linalg.norm(omega) + 1e-16)
    s = np.sin(0.5 * theta)
    return np.array([np.cos(0.5 * theta), axis[0]*s, axis[1]*s, axis[2]*s])