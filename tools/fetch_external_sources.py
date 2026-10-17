#!/usr/bin/env python3
"""Fetch external SLAM source at the recorded commits; never overwrite dirty trees."""
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DESTINATIONS={'cartographer':'third_party/cartographer_ws/src/cartographer',
 'cartographer_ros':'third_party/cartographer_ws/src/cartographer_ros',
 'liw2d':'third_party/2dliw_workspace/src/2DLIW-SLAM-main'}

def run(*args,cwd=None):return subprocess.run(args,cwd=cwd,check=True,text=True,capture_output=True).stdout.strip()

def main():
    versions=json.loads((ROOT/'third_party/versions.json').read_text())
    for name,spec in versions.items():
        destination=ROOT/DESTINATIONS[name]
        if destination.exists():
            if not (destination/'.git').is_dir():raise RuntimeError('Existing non-git directory: '+str(destination))
            if run('git','status','--porcelain',cwd=destination):raise RuntimeError('Preserving dirty source: '+str(destination))
        else:
            destination.parent.mkdir(parents=True,exist_ok=True)
            run('git','clone','--no-checkout',spec['repository'],str(destination))
        run('git','fetch','origin',spec['commit'],cwd=destination)
        run('git','checkout','--detach',spec['commit'],cwd=destination)
        assert run('git','rev-parse','HEAD',cwd=destination)==spec['commit']
        print(name,spec['commit'])

if __name__=='__main__':main()
