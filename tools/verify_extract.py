#!/usr/bin/python3
"""Verify download SHA256 and extract regular files without links/path traversal."""
import argparse
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('archive',type=Path)
parser.add_argument('--output',required=True,type=Path)
parser.add_argument('--sha256')
parser.add_argument('--bags-only',action='store_true')
args=parser.parse_args()
digest=hashlib.sha256()
with args.archive.open('rb') as stream:
    for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
value=digest.hexdigest()
if args.sha256 and value!=args.sha256:
    raise ValueError(f'Checksum mismatch: {args.archive}')
args.output.mkdir(parents=True,exist_ok=True)
paths=[]
with tarfile.open(args.archive,'r:*') as archive:
    for member in archive:
        path=Path(member.name)
        if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk():
            raise ValueError(f'Unsafe archive member: {member.name}')
        if not member.isfile() or (args.bags_only and path.suffix!='.bag'):continue
        target=args.output/path.name if args.bags_only else args.output/path
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        with archive.extractfile(member) as src, target.open('wb') as dst:
            shutil.copyfileobj(src,dst,8*1024**2)
        paths.append(str(target.resolve()))
        print(f'Extracted {target}',flush=True)
metadata={'archive':str(args.archive.resolve()),'bytes':args.archive.stat().st_size,
          'sha256':value,'expected_sha256':args.sha256,'verified_against_source':bool(args.sha256),
          'files':paths}
args.archive.with_suffix(args.archive.suffix+'.verified.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(f'Verified SHA256: {value}',flush=True)
