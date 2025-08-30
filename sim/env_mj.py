# sim/env_mj.py
import mujoco
import numpy as np
from sim.integrators import LGVI, Integrator

class MujocoEnv:
    def __init__(self, model_path, dt, integrator: Integrator | None,
                 mode='mujoco_builtin', base_body_name: str | None = None):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)
        self.dt = float(dt)
        self.t = 0.0
        self.mode = mode
        self.integrator = integrator

        # IDs for geom and "size knob" joints
        self.gid   = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,  "gbox")
        self.jid_ux = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ux_joint")
        self.jid_uy = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "uy_joint")
        self.jid_uz = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "uz_joint")

        # qpos addresses (half-extents live here)
        self.adr_ux = int(self.model.jnt_qposadr[self.jid_ux])
        self.adr_uy = int(self.model.jnt_qposadr[self.jid_uy])
        self.adr_uz = int(self.model.jnt_qposadr[self.jid_uz])
        # qvel addresses for those slide joints (use dofadr, not qposadr math)
        self.dadr_ux = int(self.model.jnt_dofadr[self.jid_ux])
        self.dadr_uy = int(self.model.jnt_dofadr[self.jid_uy])
        self.dadr_uz = int(self.model.jnt_dofadr[self.jid_uz])

        # free root body
        if base_body_name is not None:
            self.base_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        else:
            self.base_bid = self._find_free_root_body()
        assert self.base_bid >= 0, "Could not find a free root body."

        # cache initial mass/inertia (used if you keep mass constant)
        self.base_mass = float(self.model.body_mass[self.base_bid])
        J_diag = self.model.body_inertia[self.base_bid].copy()
        self.base_J = np.diag(J_diag)

        # initial sync of LGVI with model
        if isinstance(self.integrator, LGVI):
            self.integrator.m = float(self.model.body_mass[self.base_bid])
            self.integrator.J = np.diag(self.model.body_inertia[self.base_bid])

    def _find_free_root_body(self):
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return self.model.jnt_bodyid[j]
        return -1

    def pack_state(self):
        return np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()])

    def unpack_state(self, x, advance_time=False):
        nq, nv = self.model.nq, self.model.nv
        self.data.qpos[:] = x[:nq]
        self.data.qvel[:] = x[nq:nv+nq]
        mujoco.mj_forward(self.model, self.data)
        if advance_time:
            self.data.time += self.dt

    def _apply_geom_size_from_knobs(self):
        ux = float(self.data.qpos[self.adr_ux])
        uy = float(self.data.qpos[self.adr_uy])
        uz = float(self.data.qpos[self.adr_uz])
        self.model.geom_size[self.gid, 0] = ux
        self.model.geom_size[self.gid, 1] = uy
        self.model.geom_size[self.gid, 2] = uz

    def _update_body_inertia_from_size(self, mass: float | None = None):
        """Update mass/inertia to match current half-extents.
        Pass `mass=None` to keep current mass; pass a float to enforce that mass."""
        # read current half-extents
        ux = float(self.data.qpos[self.adr_ux]); a = 2.0 * ux
        uy = float(self.data.qpos[self.adr_uy]); b = 2.0 * uy
        uz = float(self.data.qpos[self.adr_uz]); c = 2.0 * uz

        # choose mass policy
        if mass is None:
            mass = float(self.model.body_mass[self.base_bid])  # keep whatever is currently set

        # box inertia about body frame principal axes
        Ix = (1.0 / 12.0) * mass * (b*b + c*c)
        Iy = (1.0 / 12.0) * mass * (a*a + c*c)
        Iz = (1.0 / 12.0) * mass * (a*a + b*b)

        self.model.body_mass[self.base_bid] = mass
        self.model.body_inertia[self.base_bid, 0] = Ix
        self.model.body_inertia[self.base_bid, 1] = Iy
        self.model.body_inertia[self.base_bid, 2] = Iz

    # ---------- wrench callback for LGVI ----------
    def dynamics_wrench(self, t, x, u):
        """Return external force/torque (world-frame) acting on the free base."""
        self.unpack_state(x)  # DOES NOT integrate

        g = np.array(self.model.opt.gravity)
        mass_live = float(self.model.body_mass[self.base_bid])   # use *live* mass
        Fg_world = mass_live * g

        # world-from-body rotation of the base
        Rwb = self.data.xmat[self.base_bid].reshape(3, 3)

        u = np.asarray(u)
        if u.size == 3:
            Fb_world = np.zeros(3)
            tau_body = u
        elif u.size == 6:
            Fb_body  = u[:3]
            tau_body = u[3:]
            Fb_world = Rwb @ Fb_body
        else:
            raise ValueError("action must be 3 (torque) or 6 (force+torque) long")

        F_world   = Fg_world + Fb_world
        tau_world = Rwb @ tau_body
        return {"F_world": F_world, "tau_world": tau_world}

    def step(self, action):
        if self.mode == 'mujoco_builtin':
            # sync dt with MuJoCo's timestep
            if abs(float(self.model.opt.timestep) - float(self.dt)) > 1e-12:
                self.dt = float(self.model.opt.timestep)

            # Position actuators target the three size joints
            self.data.ctrl[0:3] = action  # assumes order [ux, uy, uz]
            mujoco.mj_step(self.model, self.data)

            # Mirror size joints -> geom size, optionally update inertia (keep mass constant here)
            self._apply_geom_size_from_knobs()
            self._update_body_inertia_from_size(mass=self.base_mass)

            # (Not required for built-in integration, but harmless)
            if isinstance(self.integrator, LGVI):
                self.integrator.m = float(self.model.body_mass[self.base_bid])
                self.integrator.J = np.diag(self.model.body_inertia[self.base_bid])
                self.integrator.nq = int(self.model.nq)
                self.integrator.nv = int(self.model.nv)

            mujoco.mj_forward(self.model, self.data)  # refresh caches with new size/inertia
            self.t = float(self.data.time)

        else:  # 'custom' (LGVI on free base)
            # 1) Set size joints kinematically from action and zero their velocities
            self.data.qpos[self.adr_ux] = float(action[0])
            self.data.qpos[self.adr_uy] = float(action[1])
            self.data.qpos[self.adr_uz] = float(action[2])
            self.data.qvel[self.dadr_ux] = 0.0
            self.data.qvel[self.dadr_uy] = 0.0
            self.data.qvel[self.dadr_uz] = 0.0

            # 2) Mirror to geom size and (optionally) update inertia; keep mass constant here
            self._apply_geom_size_from_knobs()
            self._update_body_inertia_from_size(mass=self.base_mass)

            # 3) Sync LGVI with current model params
            if isinstance(self.integrator, LGVI):
                self.integrator.m = float(self.model.body_mass[self.base_bid])
                self.integrator.J = np.diag(self.model.body_inertia[self.base_bid])
                self.integrator.nq = int(self.model.nq)
                self.integrator.nv = int(self.model.nv)

            # 4) Recompute caches (contacts, Jacobians, etc.) at this resized state
            mujoco.mj_forward(self.model, self.data)

            # 5) Integrate *only* the free base with LGVI; size joints are held kinematic
            x = self.pack_state()
            f = lambda tt, xx: self.dynamics_wrench(tt, xx, u=np.zeros(3))  # gravity-only wrench
            self.t, x_next = self.integrator.step(f, self.t, x, self.dt)
            self.unpack_state(x_next, advance_time=True)

        obs = np.concatenate([self.data.qpos, self.data.qvel])
        rew, done, info = 0.0, False, {}
        return obs, rew, done, info

    def reset(self, qpos=None, qvel=None):
        self.data = mujoco.MjData(self.model)
        if qpos is not None:
            self.data.qpos[:] = qpos
        if qvel is not None:
            self.data.qvel[:] = qvel
        # Ensure geom size & inertia match knobs at reset
        self._apply_geom_size_from_knobs()
        self._update_body_inertia_from_size(mass=self.base_mass)
        if isinstance(self.integrator, LGVI):
            self.integrator.m = float(self.model.body_mass[self.base_bid])
            self.integrator.J = np.diag(self.model.body_inertia[self.base_bid])
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0
        return self.observe()

    def observe(self):
        return np.concatenate([self.data.qpos, self.data.qvel])
