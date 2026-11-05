"""Extend the repaired pilot, then evaluate six fixed-command scenarios sets."""
from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[1]

def main():
  subprocess.run([
    str(ROOT/'.venv/bin/go2-train'),'Go2-Velocity-Rough-AMP-Teacher',
    '--env.scene.num-envs','1024','--agent.resume','True',
    '--agent.load-run','2026-11-04_11-20-24_repaired-validation',
    '--agent.load-checkpoint','model_199.pt','--agent.max-iterations','800',
    '--agent.run-name','repaired-validation-extended',
    '--agent.logger','tensorboard','--agent.upload-model','False',
  ],cwd=ROOT,check=True)
  runs=list((ROOT/'logs/rsl_rl/go2_paper_amp_teacher').glob('*_repaired-validation-extended'))
  run=max(runs,key=lambda p:p.stat().st_mtime)
  checkpoint=max(run.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
  output=run/'validation'
  output.mkdir(exist_ok=True)
  for terrain in ('flat','rough'):
    for seed in (42,43,44):
      with (output/f'{terrain}_{seed}.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'tools/evaluate_go2_paper_amp.py'),str(checkpoint),
          '--terrain',terrain,'--seed',str(seed),'--output',str(output/f'{terrain}_{seed}.json')],
          cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
  reports=[json.loads(p.read_text()) for p in sorted(output.glob('*.json'))]
  (output/'summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2))
  print(f'Training and validation complete: {output}',flush=True)

if __name__=='__main__':
  main()
