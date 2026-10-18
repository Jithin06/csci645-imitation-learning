"""Teachers (demonstrators) for PandaPickCube.
  * ScriptedExpert - waypoint state machine + damped-least-squares IK.
"""

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco.mjx._src import support

class ScriptedExpert:
    """Pick-and-place state machine, written against the env's own action space.

    Action semantics (pick.py): ctrl <- clip(ctrl + 0.04 * a). To make the arm reach a
    desired joint vector q_des we output a = clip((ctrl_des - ctrl) / 0.04, -1, 1), where
    ctrl_des = q_des + gravity_torque / kp (the Panda's position actuators otherwise sag).

    Phases: 0 above box -> 1 descend -> 2 close fingers -> 3 lift -> 4 carry to target, hold.
    """

    APPROACH_H = 0.06    # m above the box centre for the pre-grasp waypoint
    LIFT_H = 0.08        # m lifted straight up after grasping
    REACH_TOL = 0.015    # m, waypoint reached
    GRASP_TOL = 0.008    # m, gripper site close enough to the box centre to close
    GRASP_TIMEOUT = 25   # steps in 'descend' after which 2x GRASP_TOL is accepted
    CLOSE_STEPS = 10     # control steps spent closing the fingers (0.2 s)
    FAST, SLOW = 0.02, 0.008   # m per control step (1.0 / 0.4 m/s)
    MAX_ROT = 0.03             # rad per control step
    DAMPING = 0.02
    W_ROT, W_ROT_CARRY = 0.5, 0.3   # orientation weight vs position in the IK
    MAX_TILT, TILT_FROM = 0.7, 0.70  # rad of outward pitch at x = 0.9 m; no tilt below x = 0.70 m
    LEAD = 3.0           # command this many steps ahead (the servos lag the target)

    def __init__(self, env):
        self.env = env
        m = env.mj_model
        self.site = env._gripper_site
        self.hand_body = int(m.site_bodyid[self.site])
        self.box = env._obj_body
        self.mocap = int(env._mocap_target[0])
        self.scale = float(env._action_scale)
        self.lo, self.hi = jnp.asarray(env._lowers), jnp.asarray(env._uppers)
        self.kp = jnp.asarray(m.actuator_gainprm[:7, 0])
        # Desired gripper orientation = orientation at the home keyframe (pointing down).
        import mujoco
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
        mujoco.mj_forward(m, d)
        self.R_home = jnp.asarray(d.site_xmat[self.site].reshape(3, 3))
        self._act = jax.jit(jax.vmap(self._act_single))

    def reset(self, n):
        self.phase = jnp.zeros(n, jnp.int32)
        self.count = jnp.zeros(n, jnp.int32)
        self.anchor = jnp.zeros((n, 3))          # where the grasp happened (for the lift)

    def act(self, state, obs, t):
        a, self.phase, self.count, self.anchor = self._act(state.data, self.phase, self.count, self.anchor)
        return np.asarray(a)

    def _act_single(self, data, phase, count, anchor):
        mx = self.env._mjx_model
        d = mjx.forward(mx, data)                     # make kinematics current
        g = d.site_xpos[self.site]
        R = d.site_xmat[self.site].reshape(3, 3)
        box = d.xpos[self.box]
        tgt = d.mocap_pos[self.mocap]
        up = lambda h: jnp.array([0.0, 0.0, h])

        # Far-away boxes/targets are out of reach with a vertical hand, so pitch the hand
        # outward (about world y) by up to MAX_TILT as the x-coordinate grows.
        def R_for(x):
            th = -self.MAX_TILT * jnp.clip((x - self.TILT_FROM) / (0.9 - self.TILT_FROM), 0.0, 1.0)
            c, s_ = jnp.cos(th), jnp.sin(th)
            Ry = jnp.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
            return Ry @ self.R_home
        R_grasp = R_for(box[0])
        pre = box + up(self.APPROACH_H)                   # approach straight down

        # Phase transitions (evaluated on the current state). `count` = steps spent in phase.
        d_box = jnp.linalg.norm(box - g)
        nxt = phase
        nxt = jnp.where((phase == 0) & (jnp.linalg.norm(pre - g) < self.REACH_TOL), 1, nxt)
        grasp_ok = (d_box < self.GRASP_TOL) | ((count > self.GRASP_TIMEOUT) & (d_box < 2 * self.GRASP_TOL))
        nxt = jnp.where((phase == 1) & grasp_ok, 2, nxt)
        anchor = jnp.where((phase == 1) & (nxt == 2), g, anchor)
        nxt = jnp.where((phase == 2) & (count >= self.CLOSE_STEPS), 3, nxt)
        nxt = jnp.where((phase == 3) & (jnp.linalg.norm(anchor + up(self.LIFT_H) - g) < self.REACH_TOL), 4, nxt)
        count = jnp.where(nxt == phase, count + 1, 0)

        goal = jnp.select([nxt == 0, nxt == 1, nxt == 2, nxt == 3],
                          [pre, box, anchor, anchor + up(self.LIFT_H)], tgt)
        R_des = jnp.where(nxt >= 3, R_for(tgt[0]), R_grasp)
        vmax = jnp.where((nxt == 1) | (nxt == 2), self.SLOW, self.FAST)

        # Cartesian step (speed-limited) + orientation error -> joint step (damped least squares).
        dp = goal - g
        dp = dp * jnp.minimum(1.0, vmax / (jnp.linalg.norm(dp) + 1e-9))
        drot = 0.5 * (jnp.cross(R[:, 0], R_des[:, 0]) + jnp.cross(R[:, 1], R_des[:, 1])
                      + jnp.cross(R[:, 2], R_des[:, 2]))
        jacp, jacr = support.jac(mx, d, g, self.hand_body)
        J = jnp.concatenate([jacp[:7].T, jacr[:7].T], axis=0)          # (6, 7) arm only
        w_rot = jnp.where(nxt >= 3, self.W_ROT_CARRY, self.W_ROT)  # small: lets the wrist tilt at far reach
        drot = w_rot * drot
        drot = drot * jnp.minimum(1.0, self.MAX_ROT / (jnp.linalg.norm(drot) + 1e-9))
        e = jnp.concatenate([dp, drot])
        dq = J.T @ jnp.linalg.solve(J @ J.T + self.DAMPING**2 * jnp.eye(6), e)

        q_des = d.qpos[:7] + self.LEAD * dq
        ctrl_arm = jnp.clip(q_des + d.qfrc_bias[:7] / self.kp, self.lo[:7], self.hi[:7])
        grip = jnp.where(nxt >= 2, 0.0, 0.04)                           # 0.04 open, 0 closed
        ctrl_des = jnp.concatenate([ctrl_arm, grip[None]])
        a = jnp.clip((ctrl_des - d.ctrl) / self.scale, -1.0, 1.0)
        return a, nxt, count, anchor

class NoisyPolicy:
    """The executed (noisy, clipped) action is what gets recorded when label='executed';
    label='clean' records the teacher's intended action instead (DART-style).
    """

    def __init__(self, base, std, seed=0, label="executed"):
        self.base, self.std, self.label = base, std, label
        self.rng = np.random.default_rng(seed)
        self.last_clean = None

    def reset(self, n):
        self.base.reset(n)

    def act(self, state, obs, t):
        clean = np.asarray(self.base.act(state, obs, t))
        self.last_clean = clean
        return np.clip(clean + self.rng.normal(0.0, self.std, clean.shape), -1.0, 1.0)

def make_teacher(env, kind, seed=0):
    if kind == "scripted":
        return ScriptedExpert(env)
    raise ValueError(kind)