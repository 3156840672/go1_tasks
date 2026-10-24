# Go1 Walk These Ways-inspired training task

This is a MuJoCo/mjlab port of the core behavior-conditioned locomotion idea, not
a bit-for-bit reproduction of the authors' Isaac Gym training stack.
See the [paper](https://proceedings.mlr.press/v205/margolis23a/margolis23a.pdf)
and [official training code](https://github.com/Improbable-AI/walk-these-ways)
for the original method.

## Train

```bash
uv run go1-train Go1-Velocity-MoB --env.scene.num-envs 1024 --agent.logger tensorboard
```

The default logger is Weights & Biases; omit `--agent.logger tensorboard` if you
use it.

The behavior-command sampling and reward scale were corrected after the first
MoB experiments. Start a fresh training run; checkpoints trained with the old
reward and zero-valued continuous behavior commands do not demonstrate this
configuration's behavior.

The task trains on a plane at 50 Hz (0.005 s physics step, decimation 4).
Training also randomizes body mass and inertia by approximately 10%, PD gains
by 10%, joint friction loss from 0 to 0.05, and initial joint positions and
velocities by 0.1. Existing foot friction, encoder bias, and trunk COM
randomization remain active. Every 4-8 seconds a velocity disturbance of up
to 0.3 m/s in x/y and 0.3 rad/s in yaw is applied. Play mode omits these new
disturbances and parameter randomizations. Start a fresh run to train with
the new distribution; existing checkpoints are not retroactively updated.
Training uses a reward-threshold velocity-grid curriculum. It initially samples
near-zero body-frame velocities, then increases sampling weight for neighboring
velocity cells when mean linear and yaw tracking exceed 0.8 and 0.5 over at
least one second. Curriculum weights are stored in checkpoints and restored on
resume. This is a simplified, shared grid rather than the authors' separate
gait-wise command curricula, and it uses only velocity-tracking thresholds,
not contact-reward thresholds. No terrain curriculum is used.
The actor receives a 30-step history of 57 values: angular velocity (3),
projected gravity (3), joint positions (12), joint velocities (12), previous
actions (12), velocity command (3), behavior command (8), and four phase-clock
values (4). The actor output is 12 joint-position actions before scaling. The Go1
actuators use kp=20, kd=0.5, with action scales of 0.125 for hip joints and
0.25 for thigh/calf joints. History is flattened in term-major order, so a
sim2sim controller must reproduce that exact ordering.

The 8-value behavior command is `[phase, bound, pace, frequency, body_height,
body_pitch, stance_width, swing_height]`. Four discrete gait offsets are sampled
in training; the other five values are independently sampled within ranges in
`BehaviorCommandCfg`. Those ranges are initial experimental choices, not
verified values from the paper's appendix.

## Play

```bash
uv run go1-play Go1-Velocity-MoB \
  --checkpoint-file logs/rsl_rl/go1_mob/YOUR_RUN/model_1000.pt \
  --viewer viser
```

The Viser viewer exposes gait and behavior sliders alongside the velocity
command controls. Confirm the checkpoint path under your training log directory.

## Fidelity limits

The policy uses history directly; it does not have the paper's separate
supervised body-velocity/friction estimator. This task also does not include
the learned actuator network, 20 ms action delay, exact gait-wise command
curriculum, or all of the paper's domain-randomization ranges. Reward scales
now follow the authors' `scripts/train.py` convention: mjlab applies the
0.02-second step duration to the final reward, while the exponential auxiliary
cost uses the unscaled training coefficients. Contact targets are smoothed and
thigh/calf ground contacts are penalized. Foot slip uses horizontal velocity
with an instantaneous contact flag; the authors' code also includes a
previous-contact filter. The paper's
printed phase equation and gait labels do not agree for bound and pace; this
implementation follows the named physical foot pairings (FR/FL, RR/RL for
bound; FR/RR, FL/RL for pace). Quantitative claims from the paper must be
re-evaluated here.

The gait-free comparison task has been removed from this repository. Do not
deploy this policy on physical hardware without separate system identification,
latency modeling, safety checks, and real-robot validation.
