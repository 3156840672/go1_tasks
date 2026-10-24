"""Persist MoB command curriculum weights alongside policy checkpoints."""

from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .mob_curriculum import GridAdaptiveVelocityCommand


class MobOnPolicyRunner(VelocityOnPolicyRunner):
  def save(self, path: str, infos=None) -> None:
    twist = self.env.unwrapped.command_manager.get_term("twist")
    if isinstance(twist, GridAdaptiveVelocityCommand):
      infos = {
        **(infos or {}),
        "mob_velocity_curriculum": twist.curriculum_state_dict(),
      }
    super().save(path, infos)

  def load(self, path: str, load_cfg=None, strict=True, map_location=None) -> dict:
    infos = super().load(path, load_cfg, strict, map_location)
    twist = self.env.unwrapped.command_manager.get_term("twist")
    if (
      isinstance(twist, GridAdaptiveVelocityCommand)
      and isinstance(infos, dict)
      and "mob_velocity_curriculum" in infos
    ):
      twist.load_curriculum_state_dict(infos["mob_velocity_curriculum"])
    return infos
