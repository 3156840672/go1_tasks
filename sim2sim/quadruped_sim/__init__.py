"""Shared MuJoCo sim2sim library for the Go1 and Go2 runners.

Nothing in this package imports ``mjlab`` or ``rsl_rl``: it only needs numpy,
mujoco and torch, so the runners work without the training dependencies.  The
command-line entry points live one directory up (``go1_sim.py``,
``go1_mob_sim.py``, ``go2_sim.py``); see ``sim2sim/README.md``.
"""
