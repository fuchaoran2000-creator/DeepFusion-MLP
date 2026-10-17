#!/usr/bin/python3
"""Export actual local SLAM node poses and final graph poses using upstream protos."""
import argparse
import os
import gzip
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path

SOURCE = Path(os.environ.get('CARTOGRAPHER_SOURCE', str(Path(__file__).resolve().parents[1]/'third_party/cartographer_ws/src/cartographer'))).expanduser()
ROOT = Path(__file__).resolve().parents[1]
EPOCH_TICKS = 62135596800 * 10000000


def load_proto():
    if not (SOURCE/'cartographer').is_dir():
        raise FileNotFoundError('Set CARTOGRAPHER_SOURCE to the pinned Cartographer source; see docs/ROS.md')
    generated = ROOT/'results/mentor_benchmark/nclt_diagnostic_v1/protobuf'
    generated.mkdir(parents=True,exist_ok=True)
    sources = sorted((SOURCE/'cartographer').rglob('*.proto'))
    signature = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()
    marker = generated/'source_sha256.txt'
    if not marker.exists() or marker.read_text().strip() != signature:
        subprocess.run(['protoc',f'--proto_path={SOURCE}',f'--python_out={generated}',
                        *[str(p.relative_to(SOURCE)) for p in sources]],cwd=SOURCE,check=True)
        marker.write_text(signature+'\n')
    sys.path.insert(0,str(generated))
    from cartographer.mapping.proto.serialization_pb2 import SerializationHeader,SerializedData
    return SerializationHeader,SerializedData


def records(path):
    with path.open('rb') as stream:
        if struct.unpack('<Q',stream.read(8))[0] != 0x7b1d1f7b5bf501db:
            raise ValueError('Invalid Cartographer proto stream magic')
        while True:
            size = stream.read(8)
            if not size: break
            if len(size) != 8: raise ValueError('Truncated record size')
            length = struct.unpack('<Q',size)[0]
            data = stream.read(length)
            if len(data) != length: raise ValueError('Truncated record')
            yield gzip.decompress(data)


def values(stamp,pose):
    p,q=pose.translation,pose.rotation
    return [(stamp-EPOCH_TICKS)/1e7,p.x,p.y,p.z,q.x,q.y,q.z,q.w]


def extract(path):
    Header,Data=load_proto();rows=records(path)
    header=Header();header.ParseFromString(next(rows))
    local=[];global_rows=[];constraints=[]
    for raw in rows:
        data=Data();data.ParseFromString(raw)
        kind=data.WhichOneof('data')
        if kind=='node':
            n=data.node
            if n.node_id.trajectory_id != 0: raise ValueError('Expected one trajectory')
            local.append(values(n.node_data.timestamp,n.node_data.local_pose))
        elif kind=='pose_graph':
            for trajectory in data.pose_graph.trajectory:
                if trajectory.trajectory_id != 0: raise ValueError('Expected one trajectory')
                global_rows.extend(values(n.timestamp,n.pose) for n in trajectory.node)
            constraints.extend({'submap':c.submap_id.submap_index,'node':c.node_id.node_index,
                                'tag':int(c.tag)} for c in data.pose_graph.constraint)
    for suffix,poses in [('local',local),('global',global_rows)]:
        with path.with_name(path.stem+f'_{suffix}_nodes.tum').open('w') as out:
            for row in sorted(poses):out.write(' '.join(f'{v:.9f}' for v in row)+'\n')
    stats={'format_version':header.format_version,'local_nodes':len(local),'global_nodes':len(global_rows),
           'intra_submap_constraints':sum(c['tag']==0 for c in constraints),
           'inter_submap_constraints':sum(c['tag']==1 for c in constraints),
           'constraint_caveat':'inter-submap count is not ground-truthed loop precision/recall',
           'reference_point':'tracking frame; diagnostic noimu uses base_link',
           'source_pbstream':str(path.resolve()),'upstream_proto_source':str(SOURCE)}
    path.with_name(path.stem+'_node_diagnostics.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(path,stats['local_nodes'],stats['inter_submap_constraints'],flush=True)
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('pbstreams',nargs='+',type=Path)
    for path in p.parse_args().pbstreams:extract(path)
