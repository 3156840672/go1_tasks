# Go1 RMA training

This project implements the two-stage training structure from Kumar et al.,
[*RMA: Rapid Motor Adaptation for Legged Robots*](https://www.roboticsproceedings.org/rss17/p011.html).
The implementation uses Unitree Go1 and the repository's MuJoCo/mjlab rough
terrain environment.

## Stage 1: teacher

```bash
uv run go1-train-rma teacher \
  --num-envs 128 \
  --iterations 10000 \
  --steps 32
```

The teacher jointly trains an 18-to-8 privileged environment encoder and a
policy that maps 48 proprioceptive observations plus the latent to 12 joint
actions. Checkpoints are written to
`logs/rsl_rl/go1_rma/<timestamp>/model_*.pt` by default.

## Stage 2: adaptation

```bash
uv run go1-train-rma adaptation \
  --checkpoint logs/rsl_rl/go1_rma/<timestamp>/model_<iteration>.pt \
  --num-envs 128 \
  --iterations 1000 \
  --steps 32
```

The teacher policy and encoder are frozen. The adaptation module observes the
previous 50 actor observations, predicts the teacher's 8-dimensional latent,
and controls the rollout with that prediction. Its checkpoints are saved under
the teacher run's `adaptation/` directory unless `--output` is specified.

## sim2sim

Run a trained teacher and matching adaptation checkpoint in the bundled
independent MuJoCo scene:

```bash
uv run python sim2sim/go1_sim.py \
  --teacher logs/rsl_rl/go1_rma/<run>/model_<iteration>.pt \
  --adapter logs/rsl_rl/go1_rma/<run>/adaptation/adapter_<iteration>.pt \
  --terrain rough
```

The adapter checkpoint records its teacher checkpoint and the sim2sim loader
checks that the pair is compatible.

## Differences from the paper

The paper uses Unitree A1 and RaiSim. This implementation uses Go1,
MuJoCo-Warp, the repository's rough-terrain reward and command task, and its
terrain generator. The 18 privileged values are payload mass (1), center of
mass offset (3), motor gain scale (12), foot friction (1), and local foot
clearance (1). Model parameters are sampled at environment startup. This
repository does not reproduce the paper's hardware experiments or claim its
reported performance.
