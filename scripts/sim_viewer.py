# scripts/sim_viewer.py
import os, sys, time, argparse
import numpy as np
import matplotlib.pyplot as plt

# Make 'sim/' importable when running this file directly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mujoco
try:
    from mujoco import viewer as mj_viewer
except Exception:
    import mujoco.viewer as mj_viewer

from sim.env_mj import MujocoEnv
from sim.integrators import LGVI


def parse_vec(s, n, name):
    vals = np.array([float(x) for x in s.split(",")], dtype=float)
    if vals.size != n:
        raise ValueError(f"--{name} expects {n} comma-separated numbers")
    return vals


def angular_momentum_world(env) -> np.ndarray:
    """Angular momentum of the free base about its COM, world frame."""
    J = np.diag(env.model.body_inertia[env.base_bid].copy())
    omega_b = env.data.qvel[3:6].copy()
    H_body = J @ omega_b
    Rwb = env.data.xmat[env.base_bid].reshape(3, 3)
    return Rwb @ H_body


def set_initial_state(env: MujocoEnv, pos, quat, linvel, angvel, sizes):
    """Set initial free-joint pose/vel and size joints, then make model consistent."""
    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()

    qpos[0:3] = pos
    quat = np.array(quat, dtype=float); quat /= (np.linalg.norm(quat) + 1e-16)
    qpos[3:7] = quat
    qvel[0:3] = linvel
    qvel[3:6] = angvel

    # size joints (half-extents)
    qpos[env.adr_ux], qpos[env.adr_uy], qpos[env.adr_uz] = sizes
    qvel[env.dadr_ux] = qvel[env.dadr_uy] = qvel[env.dadr_uz] = 0.0

    env.reset(qpos=qpos, qvel=qvel)  # applies size->geom + inertia


def main():
    p = argparse.ArgumentParser("Resizable box viewer (LGVI vs MuJoCo)")
    p.add_argument("--model", type=str,
                   default=os.path.join(ROOT, "models", "resizable_box.xml"))
    p.add_argument("--mode", choices=["custom", "mujoco_builtin"], default="custom")
    p.add_argument("--dt", type=float, default=0.002)

    # initial state
    p.add_argument("--pos",     type=str, default="0,0,0.6")
    p.add_argument("--quat",    type=str, default="1,0,0,0")    # w,x,y,z
    p.add_argument("--linvel",  type=str, default="0,0,0")
    p.add_argument("--angvel",  type=str, default="0,0,0")
    p.add_argument("--sizes",   type=str, default="0.30,0.25,0.40")  # ux,uy,uz

    # plotting / logging
    p.add_argument("--print-every", type=float, default=0.5)

    args = p.parse_args()

    # Build env
    integ = LGVI(m=1.0, J=np.diag([0.01, 0.01, 0.01])) if args.mode == "custom" else None
    env = MujocoEnv(model_path=args.model, dt=args.dt, integrator=integ,
                    mode=args.mode, base_body_name="box")

    # Initial state
    pos    = parse_vec(args.pos, 3, "pos")
    quat   = parse_vec(args.quat, 4, "quat")
    linvel = parse_vec(args.linvel, 3, "linvel")
    angvel = parse_vec(args.angvel, 3, "angvel")
    sizes0 = parse_vec(args.sizes, 3, "sizes")
    set_initial_state(env, pos, quat, linvel, angvel, sizes0)

    # Put initial sizes into the viewer's Control sliders (actuators ux,uy,uz)
    # This makes the sliders reflect the starting half-extents.
    if env.model.nu >= 3:
        env.data.ctrl[0:3] = sizes0

    # Angular momentum baseline
    H0 = angular_momentum_world(env)
    H0_norm = np.linalg.norm(H0)

    # Drift logging
    t_hist: list[float] = []
    drift_hist: list[float] = []

    # Realtime pacing
    t0 = time.perf_counter()
    next_print = time.time()

    print("Open the viewer 'Control' panel and move sliders 'ux', 'uy', 'uz' to resize the box.")

    try:
        with mj_viewer.launch_passive(env.model, env.data) as viewer:
            while True:
                # Read half-extents from viewer sliders (actuators ux,uy,uz)
                if env.model.nu >= 3:
                    sizes = env.data.ctrl[0:3].copy()
                else:
                    sizes = sizes0  # fallback if actuators missing

                # Step the env; both modes accept `action = half-extents`
                env.step(sizes)

                # Keep the sliders in sync with current target (useful in custom mode)
                if env.model.nu >= 3:
                    env.data.ctrl[0:3] = sizes

                # Realtime pacing: sleep if sim time is ahead of wall clock
                ahead = env.t - (time.perf_counter() - t0)
                if ahead > 0:
                    time.sleep(ahead)

                # Drift logging each step
                H   = angular_momentum_world(env)
                rel = np.linalg.norm(H - H0) / (H0_norm + 1e-12)
                t_hist.append(env.t)
                drift_hist.append(rel)

                # Light console status
                now = time.time()
                if now >= next_print:
                    ux = float(env.data.qpos[env.adr_ux])
                    uy = float(env.data.qpos[env.adr_uy])
                    uz = float(env.data.qpos[env.adr_uz])
                    print(f"t={env.t:7.3f}  size=({ux:.3f},{uy:.3f},{uz:.3f})  "
                          f"rel drift={rel:.3e}  mode={args.mode}")
                    next_print = now + max(args.print_every, env.dt)

                viewer.sync()

    except KeyboardInterrupt:
        pass
    finally:
        # Save angular-momentum drift plot
        if t_hist:
            plt.figure()
            plt.plot(t_hist, drift_hist, label=f"{args.mode}")
            plt.xlabel("time [s]");
            plt.ylabel("relative |H - H0|")
            plt.title("Angular momentum drift");
            plt.legend()
            media_dir = os.path.join(ROOT, "media")  # <— save in ../media
            os.makedirs(media_dir, exist_ok=True)
            out = os.path.join(media_dir, f"angmom_drift_{args.mode}.png")
            plt.tight_layout();
            plt.savefig(out, dpi=200)
            print(f"[saved plot] {out}")


if __name__ == "__main__":
    main()
