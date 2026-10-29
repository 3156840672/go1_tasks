# Go1 CTS training

This task implements the core concurrent teacher-student architecture from
[Wang et al., CTS (2024)](https://arxiv.org/abs/2405.10830) for the Go1 model in
mjlab. The teacher and student run in the same PPO rollout, with a fixed 3:1
environment split. Both branches share a 512-256-128 policy head and a Gaussian
action distribution. Their 32-dimensional, L2-normalized latents come from a
privileged encoder or a five-step proprioceptive-history encoder, each with
512-256 hidden units. The student encoder learns from the current teacher
latent after every PPO update. PPO gradients do not update the student encoder.

The actor observes 45 values per step: angular velocity 3, projected gravity 3,
relative joint position 12, joint velocity 12, previous action 12, and velocity
command 3. The student encoder reads a term-major flattened 225-value history.
The teacher encoder and critic receive 259 values, including the 187-point
terrain scan, base linear velocity and foot-contact measurements. The student
policy exported to ONNX has two inputs, `proprio` (45) and `history` (225), and
outputs 12 joint-position action offsets. The controller runs at 50 Hz with
Go1 PD gains Kp=20, Kd=0.5.

## Train

```bash
uv run go1-train Go1-Velocity-CTS \
  --env.scene.num-envs 256 \
  --agent.logger tensorboard
```

Use a positive multiple of four for the environment count (enforced at runtime).
256 fits a 6 GiB RTX 3060;
the default is 1024 and the paper uses 8192, both requiring more GPU memory.
The task uses
Go1 rough-terrain curriculum and a separate `go1_cts` checkpoint directory.
Existing Go1 rough/MoB checkpoints have different network shapes and cannot be
resumed into this task.

CTS training samples 10% standing, 10% heading-controlled, and 10% forward-only
commands, plus pure turning commands in 20% of the remaining worlds. Pure turns
zero the linear command and retain a signed yaw command of at least 0.1 rad/s.
Pure turns last 8–15 s; other commands retain the 3–8 s sampling interval.
The command curriculum uses the
following ranges (m/s, rad/s):

| Iteration | Forward x | Lateral y | Yaw |
| --- | --- | --- | --- |
| 0-1199 | -0.3 to 0.6 | -0.2 to 0.2 | -0.15 to 0.15 |
| 1200-3599 | -0.6 to 0.8 | -0.4 to 0.4 | -0.3 to 0.3 |
| 3600-5599 | -1.0 to 1.0 | -0.7 to 0.7 | -0.5 to 0.5 |
| 5600+ | -1.0 to 1.0 | -1.0 to 1.0 | -0.5 to 0.5 |

Earlier CTS runs only sampled forward x speeds until iteration 4800 and
small lateral speeds until then. A checkpoint from that phase cannot be
expected to track reverse or substantial lateral commands in `play`. The
revised curriculum introduces both directions earlier; it applies only to
newly started training processes. The runner restores the saved environment
step counter when resuming; check the logged `Curriculum/command_vel/*`
ranges to confirm the new stage for that checkpoint.

CTS starts at terrain level 0 and retains the terrain-distance curriculum;
higher levels are unlocked through locomotion. The reward weights below are
applied with `scale_rewards_by_dt=True` (control step 0.02 s). Zero-weight
terms remain in the configuration but do not affect optimization.
CTS training and play sample the two stair types in 55% of environments
(27.5% each), up from the base rough task's 40%. Stair height increases from
0 to 8 cm over 10 curriculum rows, with 35 cm step runs. Slopes top out at
0.6 rather than 1.0; the other terrain types remain in the mix. Only training
uses the level curriculum; play samples terrain patches without it.
CTS training and play both allocate 128 contacts per world; the base rough
task's 35-contact limit can drop contacts on difficult terrain. The CTS
runner saves every 100 iterations; other Go1 tasks retain 50.

| Reward | Weight | Meaning |
| --- | ---: | --- |
| track_linear_velocity | 6.0 | Exponential body-frame x/y tracking |
| track_angular_velocity | 3.0 | Broad angular tracking, including roll/pitch stabilization |
| track_yaw_velocity | 2.0 | Narrow yaw-only tracking (`std=0.25 rad/s`) |
| linear_velocity_error_l2 | -1.0 | Dense x/y tracking error; avoids saturation at large errors |
| upright | 0.1 | Align trunk with terrain normal (CTS only) |
| pose | 1.0 | Stay near the command-dependent nominal pose |
| dof_pos_limits | -1.0 | Joint position limits |
| action_rate_l2 | -0.1 | Action changes between control steps |
| action_acc_l2 | -0.02 | Action acceleration between control steps |
| feet_regulation | -0.05 | Paper Eq. (9): penalize horizontal foot motion close to terrain |
| foot_slip | -0.3 | Foot sliding during contact |
| standing_missing_foot_contacts | -0.3 | Per ungrounded foot when the command is nearly zero |
| soft_landing | -1e-5 | Landing contact force when a movement command exceeds 0.05 |
| self_collisions | -0.1 | Robot self-collision |
| shank_collision | -0.3 | Shank-ground collision (force above 10 N) |
| trunk_head_collision | -0.1 | Body-ground collision |
| body_ang_vel, angular_momentum | 0 | Disabled |

`feet_regulation` implements the paper's
`sum_i ||v_xy,foot,i||^2 * exp(-max(h_site,i - 0.023, 0) / (0.025 * h_des))` with
`h_des=0.278 m`, the Go1 configured nominal base height. Foot height comes
from the four-foot terrain raycast sensor. The 0.023 m subtraction converts
Go1's foot-center site height into sole clearance; negative clearances are
clipped to zero to avoid inflating the cost on terrain penetration. Unlike
the removed air-time, clearance, and landing-peak rewards, this term penalizes
horizontal foot motion close to the ground without imposing a target swing height or
fixed gait phase. It does not directly penalize vertical foot lifting while
standing still. CTS still retains foot slip, soft landing, and collision
penalties. The CTS-only standing contact cost is active only below a 0.05
combined speed/yaw command, leaving moving gaits and recovery free to lift feet.
Soft landing uses the base task's small weight and is gated on
movement command.
Existing checkpoints retain their old gait and must be fine-tuned or
retrained to learn this change. The low-speed curriculum and dense L2 cost
remain practical additions to this port, not exact copies of the paper's
reward or command distribution. The observation and action interfaces are
unchanged from earlier CTS checkpoints.

The shank-contact weight was raised from `-0.1` to `-0.3` after fixed-command
tests found persistent shank contact. This is a training-reward change, not
an inference-time correction.

A run trained without lift incentives (`2026-10-19_23-32-45_cts_contact128_full`,
`model_9800.pt`) walked with only 3.2 cm mean swing clearance (90th percentile
4.5 cm, peak 7-8 cm) at `vx=0.5 m/s`: the feet never left the ground far enough,
because `air_time` and `foot_swing_height` were disabled and the clearance and
slip costs were negligible next to tracking. Earlier revisions enabled these
terms; the current revision replaces those lift incentives with the paper's
near-ground foot regulation. Existing checkpoints keep their old gait unless
training continues with the new weights. A 50-iteration fine-tune
of `model_9800.pt` (512 envs) raised 90th-percentile swing clearance from 4.5 to
7.5 cm and the airborne fraction from 33% to 53% while holding `vx=0.52 m/s`,
but a fresh run is still recommended for the cleanest gait.
Evaluate new checkpoints with `go1-eval-cts`; its shank-contact
fractions report both instantaneous contact and contact force above 10 N at
any physics substep in each policy interval. Verify smooth, low-contact
walking before considering hardware use.

Training-time domain randomization includes link mass/inertia scaled by
0.9-1.1, an additional -0.5 to +2 kg point-mass payload at the trunk COM,
trunk COM offsets of +/-5 cm (x) and +/-3.5 cm (y/z), foot sliding friction
0.3-1.5, PD gains scaled by 0.8-1.2, and motor torque limits scaled by
0.8-1.2. The actuator samples 0-4 physics steps of command delay (0-20 ms).
These parameters are sampled once per environment at startup, except the
delay, which is sampled during stepping. Existing encoder bias and external
push randomization remain. The torque-limit range is an approximation of the
paper's motor-strength variation. Restitution is not randomized because the
current MuJoCo task has no equivalent scalar contact-restitution parameter.
All four feet share each sampled friction-axis value within an environment;
this models a common ground surface without imposing independent left/right
friction differences.
The added randomization is disabled in play mode; inherited Go1 rough
startup randomization still applies there.

The student encoder defaults to learning rate `1e-3`, five epochs and batch
size 512. At 256 environments, its 64 student worlds produce 1536 samples
per PPO iteration, or 15 student optimizer steps. Adjust these CTS-only
settings with `--agent.algorithm.student-learning-rate`,
`--agent.algorithm.student-epochs` and
`--agent.algorithm.student-batch-size`. Use the `--randomized` evaluation
switch below to compare nominal against randomized dynamics before changing
payload or COM ranges; their effect on reverse/lateral tracking is not yet
isolated.

## Play

```bash
uv run go1-play Go1-Velocity-CTS \
  --checkpoint-file logs/rsl_rl/go1_cts/YOUR_RUN/model_1000.pt \
  --viewer viser
```

Play uses the student branch only. It still constructs simulation terrain and
critic sensors, but these are not required by the exported ONNX student policy.
In Viser, enable `Commands > Twist > Enable` before setting the speed sliders.
Play retains inherited startup randomization and terrain changes to exercise
robustness; use the fixed-command evaluation below for nominal dynamics.

For a nominal flat-ground check, compare checkpoints with the same fixed
command and seed. The script removes startup domain randomization by default;
pass `--randomized` for a matched robustness check:

```bash
uv run go1-eval-cts \
  /path/to/your/model_5000.pt \
  --vx 0.5 --seconds 5
```

The script reports the commanded and observed command, actual body-frame
speed, mean absolute tracking errors, action changes, and resets. Its
`--branch teacher` option
diagnoses whether a tracking failure belongs to the whole policy or the
student encoder. A checkpoint is not deployment-ready merely because its
training reward rose: test forward, reverse, lateral, yaw, disturbances and
falls under deterministic inference. Hardware requires independent checks of
joint order, signs, default positions, command units, 50 Hz timing, 5-frame
history order, actuator gains/limits, emergency stop, and a restrained startup.

Checkpoint paths above are examples; there is no bundled CTS checkpoint.
Training resumes the saved `common_step_counter` and thus the command
curriculum stage. Inspect the logged `Curriculum/command_vel/*` values after
resuming to confirm the intended ranges. No checkpoint is approved for
hardware deployment by these configuration changes alone.
The yaw-reward and upright-weight changes affect future training only; changing
the terrain generator also changes play terrain immediately. Existing
checkpoints, including `model_8499.pt`, require further training and evaluation
before expecting better behavior.

## Differences from the paper

The paper trains on Isaac Gym with its own terrain mix, reward table, complete
privileged state, restitution randomization and shared critic input
`(state, latent)`. This task uses mjlab's Go1 rough-terrain generator and reward
terms; its critic receives privileged state alone. The paper's full
training/evaluation numbers cannot be assumed for this port. Hardware use
requires separate actuator, observation, latency and safety validation.
