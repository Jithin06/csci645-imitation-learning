"""Shared helpers for PandaPickCube imitation learning.

Everything that must be identical across teacher collection, student training and
evaluation lives here: env construction, the success definition, the batched
rollout loop, and video rendering.
"""

import json
import os
import time
from types import SimpleNamespace

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
# JAX and PyTorch share the GPU during evaluation: stop JAX grabbing 75% of memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

ENV_NAME = "PandaPickCube"
HORIZON = 150            # env default episode_length (3 s at ctrl_dt = 0.02 s)
SUCCESS_DIST = 0.05      # box centre within 5 cm of the target centre ...
SUCCESS_HOLD = 10        # ... for 10 consecutive control steps (0.2 s)
OBS_DIM, ACT_DIM = 66, 8

def make_env(impl="jax"):
    from mujoco_playground import registry
    return registry.load(ENV_NAME, config_overrides={"impl": impl})


def box_target_dist(env, data):
    """Batched distance between the box centre and the target, in metres."""
    box = data.xpos[:, env._obj_body]
    return jnp.linalg.norm(data.mocap_pos[:, env._mocap_target[0]] - box, axis=-1)

class BatchedEnv:
    """N copies of the env stepped together (vmap + jit). Arrays are numpy."""

    def __init__(self, env, num_envs):
        self.env, self.n = env, num_envs
        self._reset = jax.jit(jax.vmap(env.reset))
        self._step = jax.jit(jax.vmap(env.step))
        self._dist = jax.jit(lambda d: box_target_dist(env, d))

    def reset(self, seeds):
        keys = jnp.stack([jax.random.PRNGKey(int(s)) for s in seeds])
        self.state = self._reset(keys)
        return self.state

    def step(self, action_np):
        self.state = self._step(self.state, jnp.asarray(action_np, dtype=jnp.float32))
        return self.state

def run_episodes(env, policy, seeds, horizon=HORIZON, record=True, log_every=0):
    """Roll out `policy` on one env per seed, all in parallel.

    policy.reset(n) is called once; policy.act(state, obs_np, t) -> (n, 8) actions in [-1, 1].
    An episode ends at success (held SUCCESS_HOLD steps), out-of-bounds/NaN, or the horizon.
    Each stored action is the one chosen from the observation stored at the same index.
    """
    n = len(seeds)
    benv = BatchedEnv(env, n)
    state = benv.reset(seeds)
    policy.reset(n)

    obs_buf = np.zeros((n, horizon, OBS_DIM), np.float32)
    act_buf = np.zeros((n, horizon, ACT_DIM), np.float32)
    rew_buf = np.zeros((n, horizon), np.float32)
    dist_buf = np.zeros((n, horizon), np.float32)
    qpos_buf = np.zeros((n, horizon + 1, env.mj_model.nq), np.float32) if record else None
    qvel_buf = np.zeros((n, horizon + 1, env.mj_model.nv), np.float32) if record else None
    mocap = np.asarray(state.data.mocap_pos)  # target is fixed for the whole episode

    alive = np.ones(n, bool)
    ep_len = np.full(n, horizon, np.int32)
    success = np.zeros(n, bool)
    failed = np.zeros(n, bool)
    hold = np.zeros(n, np.int32)
    if record:
        qpos_buf[:, 0], qvel_buf[:, 0] = np.asarray(state.data.qpos), np.asarray(state.data.qvel)

    t0 = time.time()
    for t in range(horizon):
        obs = np.asarray(state.obs, np.float32)
        act = np.clip(np.asarray(policy.act(state, obs, t), np.float32), -1.0, 1.0)
        label = act if getattr(policy, "label", "executed") == "executed" else policy.last_clean
        obs_buf[alive, t], act_buf[alive, t] = obs[alive], np.asarray(label, np.float32)[alive]
        state = benv.step(act)

        rew = np.asarray(state.reward)
        dist = np.asarray(benv._dist(state.data))
        done = np.asarray(state.done) > 0
        rew_buf[alive, t], dist_buf[alive, t] = rew[alive], dist[alive]
        if record:
            qpos_buf[alive, t + 1] = np.asarray(state.data.qpos)[alive]
            qvel_buf[alive, t + 1] = np.asarray(state.data.qvel)[alive]

        hold = np.where(dist < SUCCESS_DIST, hold + 1, 0)
        newly_ok = alive & (hold >= SUCCESS_HOLD)
        newly_bad = alive & done & ~newly_ok
        success |= newly_ok
        failed |= newly_bad
        ending = newly_ok | newly_bad
        ep_len[ending] = t + 1
        alive &= ~ending
        if log_every and (t + 1) % log_every == 0:
            print(f"  t={t+1:3d}  alive={alive.sum():3d}  success={success.sum():3d}  ({time.time()-t0:.1f}s)")
        if not alive.any():
            break

    out = dict(obs=obs_buf, actions=act_buf, rewards=rew_buf, box_target_dist=dist_buf,
               ep_len=ep_len, success=success, out_of_bounds=failed,
               seeds=np.asarray(seeds, np.int64), target_pos=mocap[:, 0])
    if record:
        out.update(qpos=qpos_buf, qvel=qvel_buf, mocap_pos=mocap)
    return out

def summarize(ep):
    L = ep["ep_len"]
    return {
        "num_episodes": int(len(L)),
        "success_rate": float(ep["success"].mean()),
        "num_success": int(ep["success"].sum()),
        "out_of_bounds": int(ep["out_of_bounds"].sum()),
        "ep_len_mean": float(L.mean()), "ep_len_std": float(L.std()),
        "ep_len_success_mean": float(L[ep["success"]].mean()) if ep["success"].any() else None,
        "final_dist_mean_m": float(np.mean([ep["box_target_dist"][i, L[i] - 1] for i in range(len(L))])),
    }

def use_bundled_ffmpeg(media):
    """CARC compute nodes have no ffmpeg on PATH; use the static binary from imageio-ffmpeg."""
    import shutil
    if shutil.which("ffmpeg") is None:
        import imageio_ffmpeg
        media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())


def render_episode(env, ep, i, path, height=480, width=640, every=1):
    """Render episode i from stored qpos/qvel (works for teacher and student rollouts)."""
    import mediapy as media
    use_bundled_ffmpeg(media)
    T = int(ep["ep_len"][i])
    states = []
    for t in range(0, T + 1, every):
        d = SimpleNamespace(qpos=ep["qpos"][i, t], qvel=ep["qvel"][i, t],
                            mocap_pos=ep["mocap_pos"][i], mocap_quat=np.array([[1.0, 0, 0, 0]]),
                            xfrc_applied=np.zeros((env.mj_model.nbody, 6)))
        states.append(SimpleNamespace(data=d))
    frames = env.render(states, height=height, width=width)
    media.write_video(path, frames, fps=round(1.0 / env.dt / every))
    return frames

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))