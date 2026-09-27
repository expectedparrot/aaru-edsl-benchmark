import argparse
import os
from pathlib import Path
from .data import ROOT

def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','infer','status','archive','reproduce'])
    p.add_argument('--run',type=Path,default=ROOT/'runs/experiment')
    p.add_argument('--env',type=Path,default=ROOT/'.env')
    p.add_argument('--stage',choices=['smoke','full','retry'],default='smoke')
    p.add_argument('--allow-paid-inference',action='store_true')
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    os.environ.setdefault('EDSL_FETCH_TOKEN_PRICES','False')
    if args.action=='prepare':
        from .protocol import prepare
        prepare()
    elif args.action=='reproduce':
        from .report import reproduce
        reproduce(check=args.check)
    else:
        from . import inference
        if args.action=='infer': inference.submit(args.run,args.env,args.stage,args.allow_paid_inference)
        elif args.action=='status': inference.status(args.run,args.env)
        else: inference.archive(args.run)

if __name__=='__main__': main()
