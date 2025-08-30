import os
import sys
import time
import argparse
import numpy as np

# --- make 'sim/' importable when running as `python scripts/sim_viewer.py`
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import mujoco
import mujoco.viewer
from sim.env_mj import MujocoEnv
from sim.integrators import LGVI

def parse_args():
    p = argparse.ArgumentParser(description="Tiny MuJoCo viewer for the resizable box.")
    p.add_argument("--model", type=str, default=os.path.join(ROOT, "models", "resizable_box.xml"),
                   help="Path to MJCF model.")
    p.add_argument("--mode", type=str, default="custom",
                   choices=["custom", "mujoco_builtin"],
                   help="Use custom LGVI ('custom') or MuJoCo's integrator ('mujoco_builtin').")
    p.add_argument("--dt", type=float, default=0.002, help="Simulation dt for custom mode.")
    p.add_argument("--action", type=str, default="0.30,0.25,0.40",
                   help="Half-extents ux,uy,uz (meters). Example: 0.3,0.25,0.4")
    p.add_argument("--oscillate", action="store_true",
                   help="Animate the size with a small sinusoid.")
    p.add_argument("--freq", type=float, default=0.5,
                   help="Oscillation frequency in Hz (when --oscillate).")
    p.add_argument("--steps", type=int, default=10_000, help="Max sim steps (safety cap).")
    return p.parse_args()

def parse_action(s: str) -> np.ndarray:
    a = np.array([float(x) for x in s.split(",")], dtype=float)
    if a.shape != (3,):
        raise ValueError("--action must be 3 comma-separated numbers, e.g. 0.3,0.25,0.4")
    return a

def main():
    args = parse_args()
    base_action = parse_action(args.action)

    # Build environment
    integ = None
    if args.mode == "custom":
        # m,J will be synced from the model inside MujocoEnv.__init__
        integ = LGVI(m=1.0, J=np.diag([0.01, 0.01, 0.01]))

    env = MujocoEnv(
        model_path=args.model,
        dt=args.dt,
        integrator=integ,
        mode=args.mode,
        base_body_name="box",  # matches your XML
    )

    # Viewer loop
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            t0 = time.time()
            for k in range(args.steps):
                # Make an action (half-extents). Clamp to [0, 1] range from your XML.
                if args.oscillate:
                    phase = 2.0 * np.pi * args.freq * (time.time() - t0)
                    a = base_action * (1.0 + 0.1 * np.sin(phase))
                else:
                    a = base_action
                a = np.clip(a, 0.0, 1.0)

                env.step(a)

                # Optional: print a tiny status every ~0.5s
                if k % int(0.5 / env.dt) == 0:
                    ux, uy, uz = float(env.data.qpos[env.adr_ux]), float(env.data.qpos[env.adr_uy]), float(env.data.qpos[env.adr_uz])
                    print(f"t={env.t:7.3f}  size (ux,uy,uz)=({ux:.3f},{uy:.3f},{uz:.3f})  mode={args.mode}")

                viewer.sync()  # render and keep realtime pacing

    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
