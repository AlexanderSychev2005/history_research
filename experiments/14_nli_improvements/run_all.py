"""Run offline studies, optional model inference, then calibration; output generated locally."""
from __future__ import annotations
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
    a=argparse.ArgumentParser();a.add_argument('--bootstrap',type=int,default=100);a.add_argument('--with-inference',action='store_true')
    opts=a.parse_args()
    tasks=[('type_constraints.py',['--bootstrap',str(opts.bootstrap)]),('evidence.py',['--bootstrap',str(opts.bootstrap)])]
    if opts.with_inference:
        tasks.extend([('hard_negatives.py',['--source',s,'--infer','--bootstrap',str(opts.bootstrap)]) for s in ('gold','lela')])
    tasks.append(('calibration.py',['--bootstrap',str(opts.bootstrap)]))
    for script,args in tasks:
        cmd=[sys.executable,str(HERE/script),*args]
        print('Running:',' '.join(cmd),flush=True);subprocess.run(cmd,check=True)
    if not opts.with_inference:print('Hard-negative NLI skipped. Use hard_negatives.py --source gold/lela --infer to run model inference.')
if __name__=='__main__':main()
