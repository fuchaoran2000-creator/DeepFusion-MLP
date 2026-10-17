import unittest
import numpy as np
from causal_stream import ResidualStream, retain_offset, world, wrap
from mentor_benchmark import Sequence, corrected


class IncrementMemoryTests(unittest.TestCase):
    def norm(self):
        return dict(xm=np.zeros(32),xs=np.ones(32),ym=np.zeros(3),ys=np.ones(3),clip=np.ones(3))

    def test_constant_bias_converges_and_reject_resets_exactly(self):
        engine=ResidualStream('relative32','increment',self.norm(),lambda x:np.array([.01,0,0]),1.,integration_tau_s=2.)
        for i in range(500):
            raw=np.array([i*.1,0.,0.]);out,status=engine.step(i*.1,raw,raw)
        retention=np.exp(-.1/2)
        self.assertAlmostEqual(out[0]-raw[0],.01*retention/(1-retention),places=7)
        raw=np.array([50.,0.,0.]);out,status=engine.step(50.,raw,np.full(3,np.nan),False)
        np.testing.assert_array_equal(out,raw)
        out,status=engine.step(50.1,raw,raw)
        np.testing.assert_array_equal(out,raw)

    def test_yaw_damps_shortest_offset_at_wrap(self):
        cart=np.array([1.,2.,np.pi-.02]);candidate=np.array([2.,4.,-np.pi+.02])
        out=retain_offset(candidate,cart,.1,1.)
        self.assertAlmostEqual(wrap(out[2]-cart[2]),.04*np.exp(-.1))
        np.testing.assert_array_equal(retain_offset(candidate,cart,.1,None),candidate)

    def test_batch_stream_agree_with_gaps_variable_dt_and_origin(self):
        t=np.cumsum(np.tile([.07,.13,.1],30))
        raw=np.column_stack([4+.2*t,2+.03*t,.7+.1*t])
        available=np.ones(len(t),bool);available[22:25]=False
        prediction=np.tile([.01,-.02,.003],(len(t)-1,1))
        for tau in [None,.5,10.]:
            engine=ResidualStream('relative32','increment',self.norm(),lambda x:prediction[0],.7,integration_tau_s=tau)
            outputs=[];local_cart=[];ready=[]
            for s,p,a in zip(t,raw,available):
                out,status=engine.step(s,p,p,a);outputs.append(out);ready.append(status['ready'])
                from causal_stream import local
                local_cart.append(local(p,engine.features.origin_cart))
            cp=np.array(local_cart)
            seq=Sequence('s','scene',t,cp,cp,cp,np.ones(len(t),bool),'online',{},available,np.array(ready))
            batch=corrected(seq,prediction,.7,'increment',integration_tau_s=tau)
            batch=np.array([world(p,engine.features.origin_cart) for p in batch])
            np.testing.assert_allclose(batch,outputs,atol=1e-12,rtol=1e-12)

    def test_invalid_tau_and_pose_target_rejected(self):
        for tau in [0,-1,np.inf,np.nan]:
            with self.assertRaises(ValueError):
                ResidualStream('relative32','increment',self.norm(),lambda x:np.zeros(3),1.,integration_tau_s=tau)
            cp=np.zeros((2,3));seq=Sequence('s','scene',np.array([0.,.1]),cp,cp,cp,np.ones(2,bool),'online',{})
            with self.assertRaises(ValueError):
                corrected(seq,np.zeros((1,3)),alpha=0.,target='increment',integration_tau_s=tau)
        with self.assertRaises(ValueError):
            ResidualStream('full20','pose',self.norm(),lambda x:np.zeros(3),1.,integration_tau_s=1.)

    def test_gate_stress_uses_checkpoint_memory_for_all_modes(self):
        import torch
        from causal_stream import extract
        from gate_stress import prepare
        t=np.arange(120)*.1;cart=np.column_stack([t,np.zeros((120,2))])
        available=np.ones(120,bool)
        data=dict(t=t,cart=cart,odom=cart,available=available,hz=np.array(10.),cart_source_t=t,odom_source_t=t)
        cp,op,_,ready,_=extract(t,cart,cart,available,'relative32')
        seq=Sequence('s','room',t,cp,op,cp,available,'online',{},available,ready,'causal_grid_v1')
        state=dict(config='relative32',target='increment',integration_tau_s=2.,
                   norm={k:torch.from_numpy(v.astype(np.float32)) for k,v in self.norm().items()},
                   gate_parameters=dict(gate_alpha=1.,ood_threshold=1e8,sigma_threshold=1e8))
        def model(z):
            out=torch.zeros(len(z),6);out[:,0]=.01;return out
        work,poses,accepted,_,_,_,_=prepare(seq,data,state,model,'clean',{})
        expected=corrected(work,np.tile([float(np.float32(.01)),0,0],(119,1)),1.,'increment',integration_tau_s=2.)
        for pose in poses.values():np.testing.assert_allclose(pose,expected,atol=1e-12,rtol=1e-12)


if __name__=='__main__':unittest.main()
