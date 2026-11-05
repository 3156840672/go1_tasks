"""Fixed-command teacher evaluation with per-step errors and contact timing."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from go2_tasks import unitree_go2_paper_amp_runner_cfg
from go2_tasks.env_cfgs import unitree_go2_paper_amp_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('checkpoint', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--terrain', choices=['flat','rough'], default='flat')
  parser.add_argument('--seconds', type=float, default=20.)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--device', default='cuda:0')
  args = parser.parse_args()
  commands = torch.tensor([[0,0,0],[.3,0,0],[-.3,0,0],[0,.2,0],[0,-.2,0],[0,0,.4],[0,0,-.4]], device=args.device)
  cfg = unitree_go2_paper_amp_env_cfg()
  cfg.seed = args.seed
  cfg.scene.num_envs = len(commands)
  cfg.episode_length_s = args.seconds + 2
  cfg.curriculum = {}
  cfg.events.pop('amp_reference_state_init', None)
  cfg.events.pop('push_robot', None)
  if args.terrain == 'flat':
    cfg.scene.terrain.terrain_type = 'plane'
    cfg.scene.terrain.terrain_generator = None
  twist = cfg.commands['twist']
  twist.heading_command = False
  twist.ranges.heading = None
  twist.rel_standing_envs = twist.rel_forward_envs = twist.rel_world_envs = 0.
  twist.resampling_time_range = (1e9,1e9)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env)
  try:
    runner_cfg = asdict(unitree_go2_paper_amp_runner_cfg())
    runner_cfg['logger'] = 'tensorboard'
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    runner_cfg['actor']['bound_action_mean'] = saved.get('amp_bound_action_mean', True)
    runner = MjlabOnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=args.device)
    runner.load(str(args.checkpoint), load_cfg={'actor':True}, map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
    wrapped.reset()
    robot = env.scene['robot']
    term = env.command_manager.get_term('twist')
    alive = torch.ones(len(commands),dtype=torch.bool,device=args.device)
    sums = torch.zeros_like(commands)
    errors = torch.zeros_like(commands)
    counts = torch.zeros(len(commands),device=args.device)
    failed = torch.zeros_like(counts)
    contact_series=[]
    action_sum = torch.zeros(len(commands), 12, device=args.device)
    action_saturation = torch.zeros(len(commands), device=args.device)
    for step in range(round(args.seconds/env.step_dt)):
      term.vel_command_b.copy_(commands)
      obs = env.observation_manager.compute(update_history=True)
      with torch.inference_mode():
        torch.testing.assert_close(obs['actor'][:, -3:], commands)
        action = policy(obs)
        action_sum += action
        action_saturation += (action.abs() > 0.99).float().mean(-1)
        _, _, done, _ = wrapped.step(action)
      newly_failed = alive & done.bool()
      failed[newly_failed] = (step+1)*env.step_dt
      alive &= ~done.bool()
      if step >= round(2/env.step_dt):
        velocity = torch.cat((robot.data.root_link_lin_vel_b[:,:2], robot.data.root_link_ang_vel_b[:,2:3]),-1)
        sums += velocity*alive[:,None]
        errors += (velocity-commands).abs()*alive[:,None]
        counts += alive
        contact_series.append((env.scene['feet_ground_contact'].data.found.reshape(len(commands),-1)>0).float().cpu())
    results=[]
    for i in range(len(commands)):
      valid = counts[i].item()>0
      item={'command':commands[i].tolist(),'samples':int(counts[i]),'failed':bool(failed[i]>0),
            'survival_seconds':float(failed[i]) if failed[i]>0 else args.seconds,
            'mean_velocity':(sums[i]/counts[i]).tolist() if valid else None,
            'mean_absolute_error':(errors[i]/counts[i]).tolist() if valid else None,
            'mean_action': (action_sum[i]/round(args.seconds/env.step_dt)).tolist(),
            'action_abs_ge_0_99_fraction': float(action_saturation[i]/round(args.seconds/env.step_dt)),
            'bound_action_mean': saved.get('amp_bound_action_mean', True)}
      if valid and contact_series:
        contacts=torch.stack(contact_series)[:int(counts[i]),i]
        if contacts.shape[1]==4:
          item['diagonal_contact_agreement']=float(((contacts[:,0]==contacts[:,3]).float().mean()+(contacts[:,1]==contacts[:,2]).float().mean())/2)
          item['foot_contact_fraction']=contacts.mean(0).tolist()
      results.append(item)
    report={'checkpoint':str(args.checkpoint.resolve()),'terrain':args.terrain,'seed':args.seed,
            'seconds':args.seconds,'warmup_seconds':2,'randomization':'startup enabled; pushes disabled; normal reset', 'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
  finally:
    wrapped.close()

if __name__=='__main__':
  main()
