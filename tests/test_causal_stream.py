import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np

from causal_stream import (FeatureStream, GridSampler, ResidualStream, extract,
                           PROTOCOL, world)
from mentor_benchmark import load_sequence, features, corrected, targets


class CausalStreamTests(unittest.TestCase):
    def test_future_odom_does_not_hide_usable_past_and_stale_resets(self):
        sampler = GridSampler()
        sampler.push_odom(10., [1, 2, .3])
        sampler.push_odom(11., [100, 200, 3])
        first = sampler.push_cart(10., [2, 3, .4])[0]
        np.testing.assert_array_equal(first[2], [1, 2, .3])
        self.assertTrue(first[3])
        later = sampler.push_cart(10.3, [3, 4, .5])
        self.assertEqual([r[0] for r in later], [10.1, 10.2, 10.3])
        self.assertFalse(later[-1][3])
        self.assertEqual(later[-1][-1], 10.)

    def test_gap_resets_statistics_and_preserves_origin(self):
        stream = FeatureStream('relative32')
        for i in range(10):
            stream.step(i*.1, [4+i, 5, .7], [8+i*2, 1, -.2])
        origin = stream.origin_cart.copy()
        stream.step(1., [14, 5, .7], [np.nan]*3, False)
        self.assertEqual(len(stream.history), 0)
        _, _, _, ready = stream.step(1.1, [15, 5, .7], [30, 1, -.2])
        self.assertFalse(ready)
        _, _, x, ready = stream.step(1.2, [15.1, 5, .7], [30.2, 1, -.2])
        self.assertTrue(ready)
        np.testing.assert_array_equal(stream.origin_cart, origin)
        np.testing.assert_allclose(x[:9], x[9:18], atol=1e-7)
        np.testing.assert_array_equal(x[18:27], np.zeros(9))

    def test_increment_missing_and_restart_removes_integrated_offset(self):
        norm = {'xm':np.zeros(32), 'xs':np.ones(32), 'ym':np.zeros(3),
                'ys':np.ones(3), 'clip':np.ones(3)}
        engine = ResidualStream('relative32','increment',norm,lambda x:np.array([.01,0,0]),1.)
        for i in range(20):
            raw=np.array([4+i*.1,2,.4]);result,status=engine.step(i*.1,raw,raw)
        self.assertGreater(np.linalg.norm(result[:2]-raw[:2]),.18)
        raw=np.array([6,2,.4]);result,status=engine.step(2.,raw,[np.nan]*3,False)
        np.testing.assert_array_equal(result,raw)
        raw=np.array([6.1,2,.4]);result,status=engine.step(2.1,raw,raw)
        self.assertFalse(status['ready']);np.testing.assert_array_equal(result,raw)
        raw=np.array([6.2,2,.4]);result,status=engine.step(2.2,raw,raw)
        self.assertAlmostEqual(np.linalg.norm(result[:2]-raw[:2]),.01)

    def test_captured_training_features_and_masks_independent_of_gt(self):
        t=10+np.arange(60)*.1
        cart=np.column_stack([4+np.sin(t),7+.2*t,.1*t])
        odom=cart+np.array([2,3,.3]);available=np.ones(len(t),bool)
        available[:3]=False;available[22:27]=False;odom[22:27]=np.nan
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            np.savez(root/'stream.npz',protocol=PROTOCOL,hz=10.,t=t,cart=cart,odom=odom,available=available)
            def truth(start):
                p=cart[start:];np.savetxt(root/'gt.tum',np.column_stack([t[start:],p[:,:2],np.zeros(len(p)),
                    np.zeros((len(p),2)),np.sin(p[:,2]/2),np.cos(p[:,2]/2)]))
            truth(0)
            row={'sequence_id':'s','scene_id':'r','gt':str(root/'gt.tum'),
                 'sampled_stream':str(root/'stream.npz'),'stream_protocol':PROTOCOL}
            a=load_sequence(row);truth(10);b=load_sequence(row)
            for config in ('full20','relative32'):
                raw=extract(t,cart,odom,available,config)
                np.testing.assert_array_equal(raw[3],a.ready)
                np.testing.assert_allclose(features(a,config),raw[2][1:],atol=1e-7,rtol=1e-7)
                np.testing.assert_array_equal(features(a,config),features(b,config))
            _,valid=targets(a,'increment')
            self.assertFalse(valid[26])  # first recovered frame cannot form a target
            prediction=np.tile([.01,0,0],(len(t)-1,1))
            result=corrected(a,prediction,target='increment')
            np.testing.assert_array_equal(result[22:28],a.cart[22:28])

    def test_ros_adapter_and_batch_on_same_stream(self):
        from tools.online_ros_benchmark import UnifiedOnlineCorrection
        adapter=UnifiedOnlineCorrection(None,None,None)
        captured=[]
        # Avoid ROS/socket dependence while exercising its real grid adapter.
        import tools.online_ros_benchmark as online
        original=online.OnlineCorrection._publish
        online.OnlineCorrection._publish=lambda self,t,m,start,fall:captured.append((t,m.copy(),fall))
        try:
            for i in range(50):
                stamp=20+i*.037
                if i<18 or i>26:
                    adapter.sampler.push_odom(stamp-.01,[2+i*.02,3,.2])
                    adapter.sampler.push_odom(stamp+100,[100,200,2])
                matrix=np.eye(4);matrix[:3,3]=[4+i*.03,5,6]
                adapter.apply(stamp,matrix)
            with tempfile.TemporaryDirectory() as temp:
                path=Path(temp)/'stream.npz';adapter.save(path)
                with np.load(path) as data:
                    _,_,x,ready,_=extract(data['t'],data['cart'],data['odom'],data['available'],'relative32')
                    np.testing.assert_array_equal(x,data['features'])
                    np.testing.assert_array_equal(ready,data['ready'])
                    self.assertTrue((~data['available']).any())
                self.assertEqual(len(captured),len(adapter.records))
                for tick,matrix,fall in captured:
                    self.assertTrue(fall);self.assertEqual(matrix[2,3],6.)
        finally:
            online.OnlineCorrection._publish=original


if __name__=='__main__':unittest.main()
