"""Cross terrain types and curriculum levels; distinguish survival from traversal."""
from __future__ import annotations
import argparse
from dataclasses import asdict, fields
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.terrains import HfRandomUniformTerrainCfg
from go2_tasks import unitree_go2_paper_amp_runner_cfg
from go2_tasks.env_cfgs import unitree_go2_paper_amp_baseline_env_cfg


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('checkpoint',type=Path)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--levels',type=int,nargs='+',default=[0,2,4])
  parser.add_argument('--repeats',type=int,default=3)
  parser.add_argument('--seed',type=int,default=42)
  parser.add_argument('--speed',type=float,default=.4)
  parser.add_argument('--seconds',type=float,default=20.)
  parser.add_argument('--noise-mode',choices=['legacy','current'],default='legacy')
  parser.add_argument('--device',default='cuda:0')
  args=parser.parse_args()
  if args.repeats<1 or args.seconds<=2 or args.speed<=0:parser.error('Invalid repeats, duration or speed')
  cfg=unitree_go2_paper_amp_baseline_env_cfg()
  gen=cfg.scene.terrain.terrain_generator
  if any(level<0 or level>=gen.num_rows for level in args.levels):parser.error('Invalid terrain level')
  if args.noise_mode=='legacy':
    noise=gen.sub_terrains['random_rough']
    gen.sub_terrains['random_rough']=HfRandomUniformTerrainCfg(**{f.name:getattr(noise,f.name) for f in fields(HfRandomUniformTerrainCfg)})
  names=list(gen.sub_terrains)
  cases=[{'terrain':name,'type':i,'level':level,'repeat':rep} for i,name in enumerate(names) for level in args.levels for rep in range(args.repeats)]
  n=len(cases);cfg.scene.num_envs=n;cfg.seed=args.seed
  cfg.episode_length_s=args.seconds+2;cfg.curriculum={}
  for key in ['amp_reference_state_init','push_robot','randomize_terrain']:cfg.events.pop(key,None)
  cfg.events['reset_base'].params['pose_range'].update(x=(0.,0.),y=(0.,0.),yaw=(0.,0.))
  twist=cfg.commands['twist'];twist.heading_command=False;twist.ranges.heading=None
  twist.rel_standing_envs=twist.rel_forward_envs=twist.rel_world_envs=0.
  twist.resampling_time_range=(1e9,1e9)
  env=ManagerBasedRlEnv(cfg=cfg,device=args.device);wrapped=RslRlVecEnvWrapper(env)
  try:
    rc=asdict(unitree_go2_paper_amp_runner_cfg());rc['logger']='tensorboard'
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    rc['actor']['bound_action_mean']=saved.get('amp_bound_action_mean',True)
    runner=MjlabOnPolicyRunner(wrapped,rc,log_dir=None,device=args.device)
    runner.load(str(args.checkpoint),load_cfg={'actor':True},map_location=args.device)
    policy=runner.get_inference_policy(device=args.device)
    terrain=env.scene.terrain
    terrain.terrain_levels[:]=torch.tensor([c['level'] for c in cases],device=args.device)
    terrain.terrain_types[:]=torch.tensor([c['type'] for c in cases],device=args.device)
    terrain.env_origins[:]=terrain.terrain_origins[terrain.terrain_levels,terrain.terrain_types]
    wrapped.reset();origins=env.scene.env_origins.clone();robot=env.scene['robot'];term=env.command_manager.get_term('twist')
    command=torch.zeros(n,3,device=args.device);command[:,0]=args.speed
    active=torch.ones(n,dtype=torch.bool,device=args.device);reached=torch.zeros_like(active);failed=torch.zeros_like(active)
    times=torch.full((n,),args.seconds,device=args.device);counts=torch.zeros(n,device=args.device)
    sums=torch.zeros(n,3,device=args.device);errors=torch.zeros_like(sums);max_distance=torch.zeros(n,device=args.device)
    sensitivity=torch.zeros(n,device=args.device);scan_range=torch.zeros(n,device=args.device)
    reasons=['']*n
    for step in range(round(args.seconds/env.step_dt)):
      term.vel_command_b.copy_(command)
      obs=env.observation_manager.compute(update_history=True)
      with torch.inference_mode():
        torch.testing.assert_close(obs['actor'][:,-3:],command)
        action=policy(obs)
        if step%10==0:
          zero=dict(obs);zero['terrain']=torch.zeros_like(obs['terrain'])
          sensitivity+=(policy(zero)-action).abs().mean(-1)*active
          scan_range+=(obs['terrain'].amax(-1)-obs['terrain'].amin(-1))*active
        _,_,done,_=wrapped.step(action)
      newly_failed=active&done.bool()
      for i in newly_failed.nonzero().flatten().tolist():
        reasons[i]=','.join(name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[i]))
      failed|=newly_failed;times[newly_failed]=(step+1)*env.step_dt
      # Automatic reset invalidates the post-step root for failed environments.
      distance=torch.linalg.vector_norm(robot.data.root_link_pos_w[:,:2]-origins[:,:2],dim=-1)
      max_distance=torch.where(active&~done.bool(),torch.maximum(max_distance,distance),max_distance)
      newly_reached=active&~done.bool()&(distance>=gen.size[0]/2)
      reached|=newly_reached;times[newly_reached]=(step+1)*env.step_dt
      valid=active&~done.bool()
      if step>=round(2/env.step_dt):
        vel=torch.cat((robot.data.root_link_lin_vel_b[:,:2],robot.data.root_link_ang_vel_b[:,2:3]),-1)
        sums+=vel*valid[:,None];errors+=(vel-command).abs()*valid[:,None];counts+=valid
      active&=~(newly_failed|newly_reached)
    results=[]
    for i,c in enumerate(cases):
      samples=int(counts[i]);item=dict(c)
      item.update(reached=bool(reached[i]),failed=bool(failed[i]),seconds=float(times[i]),termination=reasons[i],max_distance=float(max_distance[i]),samples=samples,
       mean_velocity=(sums[i]/counts[i]).tolist() if samples else None,mean_absolute_error=(errors[i]/counts[i]).tolist() if samples else None,
       terrain_action_difference_sum=float(sensitivity[i]),terrain_scan_range_sum=float(scan_range[i]))
      results.append(item)
    groups=[]
    for name in names:
      for level in args.levels:
        rs=[r for r in results if r['terrain']==name and r['level']==level]
        groups.append({'terrain':name,'level':level,'reached':sum(r['reached'] for r in rs),'failed':sum(r['failed'] for r in rs),'total':len(rs),'mean_max_distance':sum(r['max_distance'] for r in rs)/len(rs)})
    report={'checkpoint':str(args.checkpoint.resolve()),'noise_mode':args.noise_mode,'seed':args.seed,'seconds':args.seconds,'speed':args.speed,'goal_distance':gen.size[0]/2,
      'conditions':'normal reset; fixed x/y/yaw; startup DR; pushes off; 3 repeats share one terrain map seed; stop scoring at first termination or tile exit', 'groups':groups,'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(groups,ensure_ascii=False,indent=2))
  finally:wrapped.close()

if __name__=='__main__':main()
