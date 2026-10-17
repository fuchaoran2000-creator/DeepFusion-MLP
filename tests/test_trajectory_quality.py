import unittest
import numpy as np
import torch
import io
import tempfile
from pathlib import Path
from unittest.mock import patch
from causal_stream import increments,add_increment
from trajectory_quality import ScanQualityBuffer,scan_descriptor,extract_quality,QualityResidualStream
from trajectory_objective import rollout,path_loss


class TrajectoryQualityTests(unittest.TestCase):
    def test_body_rollout_matches_analytical_arc_and_gradients(self):
        inc=torch.tensor([[[.1,0,.02]]*50],dtype=torch.double,requires_grad=True)
        actual=rollout(inc).detach().numpy()[0]
        p=np.zeros(3);expected=[]
        for row in inc.detach().numpy()[0]:p=add_increment(p,row);expected.append(p)
        np.testing.assert_allclose(actual,expected,atol=1e-12)
        truth=inc.detach()+torch.tensor([.01,0,-.001],dtype=torch.double)
        residual=(truth-inc.detach()).requires_grad_()
        loss=path_loss(residual,inc.detach(),truth,torch.ones(3,dtype=torch.double))
        self.assertAlmostEqual(float(loss),0.,places=20)
        perturbed=(residual.detach()+.001).requires_grad_()
        path_loss(perturbed,inc.detach(),truth,torch.ones(3,dtype=torch.double)).backward()
        self.assertTrue(torch.isfinite(perturbed.grad).all());self.assertGreater(float(perturbed.grad.abs().sum()),0.)

    def test_long_horizon_penalizes_persistent_bias(self):
        inc=torch.zeros(1,50,3);r=inc.clone();r[:,:,0]=.01
        short=path_loss(r,inc,inc,torch.ones(3),horizons=(10,))
        long=path_loss(r,inc,inc,torch.ones(3),horizons=(50,))
        self.assertAlmostEqual(float(long/short),5.,places=5)

    def test_geometry_and_future_scan_does_not_hide_past(self):
        r=np.full(1081,2.);d,last=scan_descriptor(r,-3*np.pi/4,np.pi/720,.1,30)
        self.assertEqual(last,1080);self.assertEqual(d[0],1.);self.assertEqual(d[5],1.)
        buffer=ScanQualityBuffer();buffer.push(10.,d);buffer.push(11.,d*10)
        q,stamp=buffer.at(10.1,10.09,10.08)
        self.assertEqual(stamp,10.);self.assertEqual(q[11],1.)
        np.testing.assert_array_equal(q[3:11],d)
        self.assertEqual(buffer.at(10.3,10.3,10.3)[0][11],0.)

    def test_quality_guard_is_identical_in_parameter_control(self):
        t=np.arange(20)*.1;p=np.column_stack([t,np.zeros((20,2))]);q=np.ones((20,12),np.float32)
        q[8:10,11]=0;available=np.ones(20,bool)
        a,am=extract_quality(t,p,p,available,q,'quality44')
        b,bm=extract_quality(t,p,p,available,q,'padded44')
        np.testing.assert_array_equal(am,bm);np.testing.assert_array_equal(a[:,:32],b[:,:32])
        self.assertFalse(am[8:11].any());self.assertTrue(am[11]);np.testing.assert_array_equal(b[:,32:],0.)
        altered=q.copy();altered[15:]*=100
        c,cm=extract_quality(t,p,p,available,altered,'quality44')
        np.testing.assert_array_equal(a[:15],c[:15])

    def test_runtime_removes_offset_on_missing_scan_and_restart(self):
        norm=dict(xm=np.zeros(44),xs=np.ones(44),ym=np.zeros(3),ys=np.ones(3),clip=np.ones(3))
        stream=QualityResidualStream('quality44',norm,lambda z:np.array([.01,0,0,0,0,0]),1.)
        q=np.ones(12)
        for i in range(20):out,status=stream.step(i*.1,np.array([i*.1,0,0]),np.array([i*.1,0,0]),q)
        self.assertAlmostEqual(out[0]-1.9,.19)
        q[11]=0;raw=np.array([2.,0,0]);out,status=stream.step(2.,raw,raw,q)
        np.testing.assert_array_equal(out,raw);self.assertTrue(status['fallback'])
        q[11]=1;out,status=stream.step(2.1,raw,raw,q)
        np.testing.assert_array_equal(out,raw);self.assertFalse(status['ready'])

    def test_ros_adapter_records_only_received_past_quality(self):
        from tools.online_ros_benchmark import QualityOnlineCorrection,OnlineCorrection
        adapter=QualityOnlineCorrection(None,None,io.StringIO())
        descriptor=np.arange(8,dtype=np.float32)/10
        adapter.quality_buffer.push(1.,descriptor)
        adapter.quality_buffer.push(2.,descriptor+10)
        matrix=np.eye(4)
        with patch.object(OnlineCorrection,'_publish'):
            for stamp in [1.01,1.11,1.21]:
                adapter.sampler.push_odom(stamp,np.zeros(3));adapter.apply(stamp,matrix)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'capture.npz';adapter.save(path)
            with np.load(path) as capture:
                self.assertEqual(str(capture['feature_protocol']),'causal_quality_v1')
                self.assertEqual(str(capture['quality_arrival_scope']),'actual received scan callbacks')
                np.testing.assert_array_equal(capture['quality'][:2,3:11],np.tile(descriptor,(2,1)))
                self.assertFalse(capture['ready'][2]);self.assertEqual(capture['quality'][2,11],0)
                self.assertTrue((capture['scan_source_t']<=capture['t']).all())


if __name__=='__main__':unittest.main()
