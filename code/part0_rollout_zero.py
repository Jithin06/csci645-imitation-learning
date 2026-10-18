import os
os.environ["MUJOCO_GL"] = "egl"

import jax
import jax.numpy as jnp
import mediapy as media
import imageio_ffmpeg
media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())

from mujoco_playground import registry

# Load PandaPickCube using the JAX backend.
env = registry.load(
    "PandaPickCube",
    config_overrides={"impl": "jax"},
)

# Initialize the environment.
reset_fn = jax.jit(env.reset)
step_fn = jax.jit(env.step)

state = reset_fn(jax.random.PRNGKey(0))
trajectory = [state]

rng = jax.random.PRNGKey(42)
action = jnp.zeros(env.action_size)

# Run the simulation with random actions.
for i in range(200):
    if False:  # zero actions every step
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(
            action_rng,
            shape=(env.action_size,),
            minval=-1.0,
            maxval=1.0,
        )

    state = step_fn(state, action)
    trajectory.append(state)

# Render the simulation.
print("Rendering video...")
frames = env.render(trajectory[::2], height=480, width=640)

# Save as an MP4.
media.write_video(
    "panda_pick_cube_zero.mp4",
    frames,
    fps=round(1.0 / env.dt / 2),
)

print("Saved panda_pick_cube_zero.mp4")
