"""Go2 sim2sim model and inference contract tests."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from quadruped_sim.joystick import Joystick
from quadruped_sim.policies.actor import ACTION_DIM, StandardActor
from quadruped_sim.policies.go2 import (
  HIM_OBS_DIM,
  HimActor,
  PaperAmpStudentActor,
  load_him_policy,
)
from quadruped_sim.robots.go2 import (
  ACTION_SCALE,
  AMP_OBS_DIM,
  BASE_HEIGHT,
  DEFAULT,
  EFFORT,
  JOINTS,
  KD,
  KP,
  OBS_DIM,
  SATURATION_EFFORT,
  SHADOW_RADIUS,
  VELOCITY_LIMIT,
  VISUAL_MEANSIZE,
  Go2Simulator,
  make_model,
)
from quadruped_sim.terrain import (
  CURRICULUM_DIR,
  curriculum_columns,
  curriculum_terrain_types,
)
from quadruped_sim.viewer import chase_azimuth, configure_viewer_rendering


class Go2SimTests(unittest.TestCase):
  def test_model_actuator_contract(self):
    model = make_model("flat")
    self.assertEqual(model.nu, ACTION_DIM)
    self.assertEqual(len(JOINTS), ACTION_DIM)
    for index, name in enumerate(JOINTS):
      actuator = model.actuator(name + "_motor")
      np.testing.assert_allclose(actuator.ctrlrange, [-EFFORT[index], EFFORT[index]])

  def test_dc_motor_torque_speed_limit(self):
    sim = Go2Simulator(None)
    sim.target = DEFAULT + 10.0
    sim.physics_step()
    np.testing.assert_allclose(sim.data.ctrl[sim.aid], EFFORT)

    sim.reset()
    sim.target = DEFAULT + 10.0
    sim.data.qvel[sim.vadr] = 30.0
    sim.physics_step()
    np.testing.assert_allclose(sim.data.ctrl[sim.aid], 0.0, atol=1e-6)

  def test_terrain_modes(self):
    for terrain, expected_type in (
      ("flat", mujoco.mjtGeom.mjGEOM_PLANE),
      ("rough", mujoco.mjtGeom.mjGEOM_HFIELD),
      ("stairs", mujoco.mjtGeom.mjGEOM_HFIELD),
    ):
      model = make_model(terrain, terrain_seed=7)
      ground_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")
      self.assertGreaterEqual(ground_id, 0)
      self.assertEqual(model.geom_type[ground_id], expected_type)

  def test_curriculum_terrain_mode(self):
    exported = curriculum_terrain_types()
    self.assertIn("flat", exported)
    self.assertIn("perlin_noise", exported)
    sim = Go2Simulator(None, terrain="go2_rough", terrain_type="flat", terrain_level=4)
    # Flat column: every level spawns at floor height, one patch apart.
    self.assertAlmostEqual(sim.spawn[2], 0.0)
    self.assertAlmostEqual(sim.spawn[0], -4.0)
    self.assertAlmostEqual(sim.data.qpos[2], BASE_HEIGHT)
    spawn_site = mujoco.mj_name2id(
      sim.model, mujoco.mjtObj.mjOBJ_SITE, "spawn_level_9"
    )
    self.assertGreaterEqual(spawn_site, 0)

  def test_all_terrains_scene_is_one_level(self):
    scene = CURRICULUM_DIR / "level_9.xml"
    if not scene.exists():
      self.skipTest("level scenes not exported")
    sim = Go2Simulator(
      None,
      terrain="go2_rough",
      terrain_type="all",
      terrain_column="open_stairs",
      terrain_level=9,
    )
    site = mujoco.mj_name2id(
      sim.model, mujoco.mjtObj.mjOBJ_SITE, "spawn_open_stairs"
    )
    self.assertGreaterEqual(site, 0)
    np.testing.assert_allclose(sim.spawn, sim.model.site_pos[site])
    self.assertGreater(sim.spawn[2], 0.3)  # the stairs column is elevated
    self.assertGreater(abs(sim.spawn[1]), 1.0)  # columns are offset along Y
    # One level only: every spawn patch sits at the same x.
    columns = curriculum_columns()
    positions = np.array(
      [
        sim.model.site_pos[
          mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_SITE, f"spawn_{c}")
        ][:2]
        for c in columns
      ]
    )
    # Packed in a rectangle: four distinct rows, four distinct columns.
    self.assertEqual(len({round(value, 3) for value in positions[:, 0]}), 4)
    self.assertEqual(len({round(value, 3) for value in positions[:, 1]}), 4)
    # 14 terrains in 4 x 4 cells, so the block is 32 m x 32 m, not a strip.
    self.assertLessEqual(np.abs(positions).max(), 12.0)

  def test_patch_seams_meet_the_neighbour_ground(self):
    """No patch may present a wall at its border; `tilted_grid` used to be 0.2 m up."""
    sim = Go2Simulator(None, terrain="go2_rough", terrain_type="all", terrain_level=6)
    sim.model.vis.global_.offwidth = sim.model.vis.global_.offheight = 64
    data = mujoco.MjData(sim.model)
    data.qpos[:] = sim.model.qpos0
    mujoco.mj_forward(sim.model, data)
    columns = curriculum_columns()
    for index, name in enumerate(columns):
      row, col = divmod(index, 4)
      centre = ((row - 1.5) * 8.0, (col - 1.5) * 8.0)
      heights = []
      for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        start = np.array(
          [
            centre[0] + 3.95 * np.cos(angle),
            centre[1] + 3.95 * np.sin(angle),
            4.0,
          ]
        )
        geomid = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
          sim.model,
          data,
          start,
          np.array([0.0, 0.0, -1.0]),
          None,
          1,
          -1,
          geomid,
        )
        heights.append(4.0 - distance)
      median = float(np.median(heights))
      self.assertLess(abs(median), 0.12, f"{name} rim sits at {median:+.2f} m")

  def test_level_scenes_are_compact(self):
    for level in (0, 5, 9):
      path = CURRICULUM_DIR / f"level_{level}.xml"
      if not path.exists():
        self.skipTest("level scenes not exported")
      self.assertLess(path.stat().st_size, 1.0e6)

  def test_curriculum_terrain_spawns_on_the_patch(self):
    sim = Go2Simulator(
      None, terrain="go2_rough", terrain_type="open_stairs", terrain_level=9
    )
    # Level 9 of the exported stairs column is several steps above the floor.
    self.assertGreater(sim.spawn[2], 0.3)
    self.assertAlmostEqual(sim.data.qpos[2], sim.spawn[2] + BASE_HEIGHT)
    with self.assertRaises(ValueError):
      Go2Simulator(None, terrain="go2_rough", terrain_type="not_a_terrain")

  def test_camera_clip_planes_stay_in_metres(self):
    # A wide terrain map must not push the near plane onto the robot.
    model = make_model("go2_rough", 42, "all", 9)
    self.assertAlmostEqual(model.stat.extent * model.vis.map.znear, 0.2, places=4)
    self.assertAlmostEqual(model.stat.extent * model.vis.map.zfar, 80.0, places=1)

  def test_rough_diagnostic_glyphs_match_the_robot_scale(self):
    flat = Go2Simulator()
    options = mujoco.MjvOption()
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
    options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    camera = mujoco.MjvCamera()

    def diagnostic_sizes(sim):
      for _ in range(40):
        sim.physics_step()
      scene = mujoco.MjvScene(sim.model, maxgeom=10000)
      mujoco.mjv_updateScene(
        sim.model, sim.data, options, None, camera,
        mujoco.mjtCatBit.mjCAT_ALL, scene,
      )
      return np.array([
        geom.size.copy() for geom in scene.geoms[:scene.ngeom]
        if geom.category == mujoco.mjtCatBit.mjCAT_DECOR
      ])

    expected = diagnostic_sizes(flat)
    self.assertAlmostEqual(flat.model.stat.meansize, VISUAL_MEANSIZE)
    for terrain_type in ("flat", "all"):
      sim = Go2Simulator(terrain="go2_rough", terrain_type=terrain_type)
      self.assertAlmostEqual(sim.model.stat.meansize, flat.model.stat.meansize)
      self.assertAlmostEqual(sim.model.vis.map.force, flat.model.vis.map.force)
      np.testing.assert_allclose(diagnostic_sizes(sim), expected, rtol=1e-5)
      np.testing.assert_allclose(sim.data.qpos[2:], flat.data.qpos[2:], atol=1e-6)
      np.testing.assert_allclose(sim.data.qvel, flat.data.qvel, atol=1e-6)

    viewer = SimpleNamespace(
      user_scn=mujoco.MjvScene(flat.model, maxgeom=100), opt=options,
    )
    configure_viewer_rendering(viewer, False)
    self.assertEqual(viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW], 0)
    self.assertEqual(viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE], 1)

  def test_rough_shadow_projection_follows_the_robot(self):
    sim = Go2Simulator(terrain="go2_rough", terrain_type="flat", terrain_level=4)
    self.assertTrue(sim.shadows_enabled)
    self.assertAlmostEqual(
      sim.model.vis.map.shadowclip * sim.model.stat.extent, SHADOW_RADIUS, places=5
    )
    np.testing.assert_allclose(sim.data.light_xpos[0, :2], sim.data.qpos[:2])
    sim.data.qpos[0] += 1.0
    position = sim.data.qpos[:2].copy()
    sim.physics_step()
    np.testing.assert_allclose(sim.data.light_xpos[0, :2], position, atol=1e-6)

  def test_chase_camera_stays_behind_the_robot(self):
    for yaw_deg, expected in ((0.0, 180.0), (90.0, 270.0), (-90.0, 90.0)):
      half = np.radians(yaw_deg) / 2.0
      quat = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])
      self.assertAlmostEqual(chase_azimuth(quat), expected, places=5)

  def test_scene_observation_and_passive_stand(self):
    sim = Go2Simulator(None)
    self.assertEqual(tuple(sim.data.qpos[sim.qadr]), tuple(DEFAULT))
    self.assertAlmostEqual(sim.data.qpos[2], BASE_HEIGHT)
    expected = np.zeros(OBS_DIM)
    expected[6:9] = [0, 0, -1]
    np.testing.assert_allclose(sim.observation(), expected, atol=1e-6)
    for _ in range(800):
      sim.physics_step()
    self.assertTrue(np.isfinite(sim.data.qpos).all())
    self.assertGreater(sim.data.qpos[2], 0.15)
    self.assertGreater(sim.data.qpos[3], 0.98)
    self.assertLess(abs(sim.data.qpos[4]), 0.2)
    self.assertLess(abs(sim.data.qpos[5]), 0.2)

  def test_policy_action_contract(self):
    sim = Go2Simulator(None)
    sim.policy = StandardActor((64, 64)).eval()
    sim.set_active(True)
    sim.command[:] = [0.3, 0.0, 0.1]
    sim.policy_step()
    self.assertEqual(sim.observation().shape, (OBS_DIM,))
    np.testing.assert_allclose(
      sim.target, DEFAULT + ACTION_SCALE * sim.last_action
    )
    np.testing.assert_allclose(sim.observation()[-3:], sim.command)
    sim.set_active(False)
    sim.policy_step()
    np.testing.assert_allclose(sim.target, DEFAULT)
    self.assertFalse(sim.last_action.any())

  def test_amp_policy_observation_contract(self):
    sim = Go2Simulator(None)
    sim.policy = StandardActor((512, 256, 128), obs_dim=AMP_OBS_DIM).eval()
    sim.set_active(True)
    sim.command[:] = [0.3, -0.2, 0.7]
    obs = sim.observation()
    self.assertEqual(obs.shape, (AMP_OBS_DIM,))
    np.testing.assert_allclose(obs[:3], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(obs[3:6], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[-3:], sim.command)
    sim.policy_step()
    self.assertEqual(sim.last_action.shape, (ACTION_DIM,))

  def test_paper_amp_lstm_state_resets(self):
    sim = Go2Simulator(None)
    sim.policy = PaperAmpStudentActor().eval()
    sim.set_active(True)
    sim.policy_step()
    self.assertIsNotNone(sim.policy.hidden)
    self.assertEqual(sim.policy.hidden[0].shape, (1, 1, 256))
    sim.reset()
    self.assertTrue(torch.count_nonzero(sim.policy.hidden[0]) == 0)

  def test_him_policy_history_contract(self):
    policy = HimActor().eval()
    sim = Go2Simulator(None)
    sim.policy = policy
    sim.set_active(True)
    sim.command[:] = [0.3, -0.2, 0.7]
    sim.data.qvel[sim.vadr] = 1.0
    obs = sim.observation()
    self.assertEqual(obs.shape, (HIM_OBS_DIM,))
    np.testing.assert_allclose(obs[:3], sim.command * [2.0, 2.0, 0.25])
    np.testing.assert_allclose(obs[3:6], np.zeros(3), atol=1e-6)
    np.testing.assert_allclose(obs[6:9], [0, 0, -1], atol=1e-6)
    np.testing.assert_allclose(obs[21:33], np.full(12, 0.05), atol=1e-6)
    sim.policy_step()
    self.assertEqual(sim.last_action.shape, (ACTION_DIM,))
    # The module keeps a newest-first ring buffer of exactly this observation.
    np.testing.assert_allclose(policy._history[0, 0].numpy(), obs, atol=1e-6)
    for _ in range(10):
      sim.policy_step()
    self.assertTrue(torch.isfinite(policy._history).all())
    sim.reset()
    self.assertEqual(
      torch.count_nonzero(policy._history).item(), 0
    )
    np.testing.assert_allclose(sim.target, DEFAULT)

  def test_old_him_checkpoint_is_rejected(self):
    with TemporaryDirectory() as directory:
      path = Path(directory) / "old_him.pt"
      torch.save({"actor_state_dict": {
        "estimator.encoder.0.weight": torch.zeros(128, 270)
      }}, path)
      with self.assertRaisesRegex(ValueError, "Incompatible HIM checkpoint"):
        load_him_policy(path)

  def test_non_finite_policy_is_rejected(self):
    class BadPolicy:
      def __call__(self, obs):
        return torch.full((obs.shape[0], ACTION_DIM), float("nan"))

    sim = Go2Simulator(None)
    sim.policy = BadPolicy()
    sim.set_active(True)
    with self.assertRaises(FloatingPointError):
      sim.policy_step()

  def test_joystick_is_shared_with_go1(self):
    joystick = Joystick("")
    try:
      self.assertIsNone(joystick.command())
    finally:
      joystick.close()

  def test_constants_match_training_contract(self):
    from go2_tasks.deployment import (
      ACTION_SCALE as DEPLOY_ACTION_SCALE,
      DEFAULT_JOINT_POS,
      JOINT_LOWER,
      JOINT_NAMES,
      JOINT_UPPER,
      KD as DEPLOY_KD,
      KP as DEPLOY_KP,
      SATURATION_EFFORT as DEPLOY_SATURATION_EFFORT,
      TORQUE_LIMIT,
      VELOCITY_LIMIT as DEPLOY_VELOCITY_LIMIT,
    )

    self.assertEqual(JOINTS, JOINT_NAMES)
    np.testing.assert_allclose(ACTION_SCALE, DEPLOY_ACTION_SCALE)
    np.testing.assert_allclose(DEFAULT, DEFAULT_JOINT_POS)
    np.testing.assert_allclose(EFFORT, TORQUE_LIMIT)
    np.testing.assert_allclose(KP, DEPLOY_KP)
    np.testing.assert_allclose(KD, DEPLOY_KD)
    np.testing.assert_allclose(SATURATION_EFFORT, DEPLOY_SATURATION_EFFORT)
    np.testing.assert_allclose(VELOCITY_LIMIT, DEPLOY_VELOCITY_LIMIT)
    model = make_model("flat")
    for name, lower, upper in zip(JOINT_NAMES, JOINT_LOWER, JOINT_UPPER):
      np.testing.assert_allclose(model.joint(name).range, [lower, upper], atol=1e-5)


if __name__ == "__main__":
  unittest.main()
