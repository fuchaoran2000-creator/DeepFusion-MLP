import gzip
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from tools.extract_pbstream_diagnostics import load_proto,records,extract,EPOCH_TICKS,SOURCE


class PbstreamDiagnosticTests(unittest.TestCase):
    @unittest.skipUnless((SOURCE/'cartographer').is_dir() and shutil.which('protoc'), 'Optional: configure CARTOGRAPHER_SOURCE and protoc')
    def test_actual_proto_fields_frames_and_unix_epoch(self):
        Header,Data=load_proto();header=Header(format_version=2)
        graph=Data();trajectory=graph.pose_graph.trajectory.add();trajectory.trajectory_id=0
        gn=trajectory.node.add();gn.timestamp=EPOCH_TICKS+12345678901234567
        gn.pose.translation.x=7.;gn.pose.rotation.w=1.
        for tag in [0,1]:graph.pose_graph.constraint.add().tag=tag
        node=Data();node.node.node_data.timestamp=gn.timestamp
        node.node.node_data.local_pose.translation.y=3.;node.node.node_data.local_pose.rotation.w=1.
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'test.pbstream'
            with path.open('wb') as stream:
                stream.write(struct.pack('<Q',0x7b1d1f7b5bf501db))
                for msg in [header,graph,node]:
                    compressed=gzip.compress(msg.SerializeToString())
                    stream.write(struct.pack('<Q',len(compressed)));stream.write(compressed)
            stats=extract(path)
            self.assertEqual(stats['local_nodes'],1);self.assertEqual(stats['inter_submap_constraints'],1)
            lp=np.loadtxt(path.with_name('test_local_nodes.tum'));gp=np.loadtxt(path.with_name('test_global_nodes.tum'))
            self.assertAlmostEqual(lp[0],1234567890.1234567)
            self.assertEqual(lp[2],3.);self.assertEqual(gp[1],7.)
            np.testing.assert_array_equal(lp[4:],[0,0,0,1])

    def test_truncated_record_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.pbstream'
            path.write_bytes(struct.pack('<QQ',0x7b1d1f7b5bf501db,100)+b'abc')
            with self.assertRaises(ValueError):list(records(path))

    def test_invalid_header_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'bad.pbstream';path.write_bytes(struct.pack('<Q',42))
            with self.assertRaises(ValueError):list(records(path))


if __name__=='__main__':unittest.main()
