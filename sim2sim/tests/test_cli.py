"""Shared command-line and terrain contract of the three runners."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import go1_mob_sim
import go1_sim
import go2_sim

from quadruped_sim.cli import (
  CAMERA_MODES,
  TERRAIN_LEVELS,
  HeadlessResult,
  check_command,
  check_common,
  simulate,
)
from quadruped_sim.robots.go1 import BASE_HEIGHT as GO1_HEIGHT
from quadruped_sim.robots.go1 import Go1Simulator
from quadruped_sim.robots.go1_mob import MobSimulator
from quadruped_sim.robots.go2 import BASE_HEIGHT as GO2_HEIGHT
from quadruped_sim.robots.go2 import Go2Simulator
from quadruped_sim.scene import EXPORTED_TERRAIN
from quadruped_sim.terrain import (
  TERRAIN_MODES,
  curriculum_columns,
  curriculum_terrain_types,
)

PARSERS = {
  "go1": go1_sim.build_parser,
  "mob": go1_mob_sim.build_parser,
  "go2": go2_sim.build_parser,
}


def build_parser(name: str):
  return PARSERS[name]()


SHARED_ARGS = (
  "--headless",
  "--steps",
  "12",
  "--command",
  "0.2",
  "-0.1",
  "0.3",
  "--terrain",
  EXPORTED_TERRAIN,
  "--terrain-type",
  "all",
  "--terrain-column",
  "flat",
  "--terrain-level",
  "3",
  "--camera",
  "chase",
  "--follow-distance",
  "3.0",
  "--no-shadows",
  "--joystick",
  "",
)


def action(parser, flag):
  for item in parser._actions:
    if flag in item.option_strings:
      return item
  raise AssertionError(f"Missing flag {flag}")


class SharedFlagTests(unittest.TestCase):
  def test_every_runner_offers_the_same_terrain_and_view_flags(self):
    for name, build in PARSERS.items():
      parser = build()
      with self.subTest(runner=name):
        self.assertEqual(tuple(action(parser, "--terrain").choices), TERRAIN_MODES)
        self.assertEqual(
          tuple(action(parser, "--terrain-type").choices),
          curriculum_terrain_types(),
        )
        self.assertEqual(
          tuple(action(parser, "--terrain-column").choices), curriculum_columns()
        )
        self.assertEqual(
          tuple(action(parser, "--terrain-level").choices), TERRAIN_LEVELS
        )
        self.assertEqual(tuple(action(parser, "--camera").choices), CAMERA_MODES)
        for flag in ("--headless", "--steps", "--no-shadows", "--joystick"):
          action(parser, flag)

  def test_the_same_command_line_parses_for_every_runner(self):
    for name, build in PARSERS.items():
      args = build().parse_args(SHARED_ARGS)
      with self.subTest(runner=name):
        self.assertTrue(args.headless)
        self.assertEqual(args.steps, 12)
        self.assertEqual(tuple(args.command), (0.2, -0.1, 0.3))
        self.assertEqual(args.terrain, EXPORTED_TERRAIN)
        self.assertEqual(args.terrain_type, "all")
        self.assertEqual(args.terrain_column, "flat")
        self.assertEqual(args.terrain_level, 3)
        self.assertEqual(args.camera, "chase")
        self.assertAlmostEqual(args.follow_distance, 3.0)
        self.assertTrue(args.no_shadows)
        self.assertEqual(args.joystick, "")

  def test_default_terrain_differs_only_by_robot(self):
    expected = {"go1": "rough", "mob": "flat", "go2": "flat"}
    for name, build in PARSERS.items():
      self.assertEqual(build().parse_args([]).terrain, expected[name])

  def test_command_limits_match_the_training_task(self):
    limits = {
      "go1": go1_sim.COMMAND_LIMITS,
      "mob": go1_mob_sim.COMMAND_LIMITS,
      "go2": tuple(float(value) for value in go2_sim.COMMAND_LIMITS),
    }
    self.assertEqual(tuple(limits["go1"]), (1.0, 1.0, 0.5))
    self.assertEqual(tuple(limits["mob"]), (1.0, 0.6, 1.0))
    np.testing.assert_allclose(limits["go2"], (1.0, 0.6, 1.0))
    for name, limits_value in limits.items():
      with self.subTest(runner=name):
        parser = build_parser(name)
        with self.assertRaises(SystemExit):
          check_command(parser, (9.0, 9.0, 9.0), limits_value)
        check_command(parser, (0.5, 0.0, 0.0), limits_value)

  def test_steps_must_be_positive(self):
    for name, build in PARSERS.items():
      with self.subTest(runner=name):
        with self.assertRaises(SystemExit):
          check_common(build(), build().parse_args(["--steps", "0"]))

  def test_mob_checkpoint_flag_is_a_legacy_alias(self):
    args = go1_mob_sim.build_parser().parse_args(["--checkpoint", "mob.pt"])
    self.assertEqual(args.policy, Path("mob.pt"))

  def test_mob_can_start_without_a_checkpoint(self):
    args = go1_mob_sim.build_parser().parse_args([])
    self.assertIsNone(args.policy)

  def test_cts_flag_is_accepted_by_the_go1_and_go2_runners(self):
    for name in ("go1", "go2"):
      with self.subTest(runner=name):
        args = build_parser(name).parse_args(["--cts", "run.onnx"])
        self.assertEqual(args.cts, Path("run.onnx"))
    self.assertNotIn("--cts", go1_mob_sim.build_parser().format_help())

  def test_two_policy_sources_are_rejected(self):
    cases = {
      "go1": (go1_sim, ["--policy", "a.pt", "--cts", "b.onnx"]),
      "go2": (go2_sim, ["--policy", "a.pt", "--cts", "b.onnx"]),
    }
    for name, (module, argv) in cases.items():
      with self.subTest(runner=name):
        with mock.patch.object(sys, "argv", ["runner", *argv]):
          with self.assertRaises(SystemExit):
            module.main()


class HeadlessResultTests(unittest.TestCase):
  def test_format_is_identical_for_every_runner(self):
    result = HeadlessResult(100, 0.5, (0.4, 0.0, -0.1))
    self.assertEqual(
      result.format((0.5, 0.0, 0.0)),
      "Completed 100 steps; height=0.500 m; mean vx/vy/wz=+0.400/+0.000/-0.100; "
      "abs error=0.100/0.000/0.100",
    )

  def test_missing_velocity_keeps_the_height_only_line(self):
    self.assertEqual(
      HeadlessResult(60, 0.266, None).format((0.0, 0.0, 0.0)),
      "Completed 60 steps; height=0.266 m",
    )


class ExportedTerrainTests(unittest.TestCase):
  """The exported map must behave the same whichever robot stands on it."""

  def setUp(self):
    options = {
      "terrain": EXPORTED_TERRAIN,
      "terrain_seed": 42,
      "terrain_type": "all",
      "terrain_level": 2,
      "terrain_column": "wave_terrain",
    }
    self.go1 = Go1Simulator(None, None, **options)
    self.mob = MobSimulator(None, **options)
    self.go2 = Go2Simulator(None, **options)
    self.sims = (self.go1, self.mob, self.go2)

  def test_every_robot_spawns_on_the_same_cell(self):
    for sim in self.sims:
      np.testing.assert_allclose(sim.data.qpos[:2], sim.spawn[:2])
      self.assertGreaterEqual(sim.light_mocap_id, 0)
      np.testing.assert_allclose(sim.spawn, self.go2.spawn)

  def test_standing_height_uses_each_robots_own_base_height(self):
    self.assertAlmostEqual(
      self.go1.data.qpos[2], self.go1.spawn[2] + GO1_HEIGHT, places=6
    )
    self.assertAlmostEqual(
      self.mob.data.qpos[2], self.mob.spawn[2] + GO1_HEIGHT, places=6
    )
    self.assertAlmostEqual(
      self.go2.data.qpos[2], self.go2.spawn[2] + GO2_HEIGHT, places=6
    )

  def test_headless_sampling_works_for_every_robot(self):
    for sim in self.sims:
      result = simulate(sim, 40)
      self.assertIsNotNone(result.mean)
      self.assertIn("abs error=", result.format(sim.command))


if __name__ == "__main__":
  unittest.main()
