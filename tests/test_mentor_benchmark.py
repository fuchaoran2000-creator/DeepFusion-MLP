import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mentor_benchmark import (Sequence, CONFIGS, features, corrected, metrics,
                              validate_manifest, load_sequence, loss, causal_windows)
from portable_residual import motion_feature, PortableResidual
from residual_models import Regressor
from tools.summarize_mentor_benchmark import rigid_alignment, shared_anchor_alignment


def sequence():
    t=np.arange(100)*.1
    cart=np.column_stack([t,np.zeros_like(t),np.zeros_like(t)])
    return Sequence('synthetic','room',t,cart,cart.copy(),cart.copy(),
                    np.ones(len(t),dtype=bool),'online',{})


class BenchmarkTests(unittest.TestCase):
    def test_reject_sequence_and_scene_leakage(self):
        manifest={'train':['a'],'validation':['b'],'test':['c'],'isolation_field':'scene_id',
                  'sequences':[{'sequence_id':n,'scene_id':scene,'cart':n+'.tum'}
                               for n,scene in [('a','room1'),('b','room2'),('c','room1')]]}
        with self.assertRaises(ValueError): validate_manifest(manifest)
        manifest['sequences'][2]['scene_id']='room3'
        validate_manifest(manifest)
        manifest['test']=['a']
        with self.assertRaises(ValueError): validate_manifest(manifest)

    def test_feature_dimensions_and_no_reference_dependency(self):
        seq=sequence()
        expected={'full20':20,'no_P_C':16,'no_P_O':16,'no_D_C':16,'no_D_O':16,
                  'no_increments':12,'no_E':16,'poses8':8,'motion_difference12':12,
                  'relative6':6,'relative9':9,'relative14':14,'relative32':32,
                  'no_window_mean':23,'no_window_std':23,'no_motion_state':27}
        altered=copy.deepcopy(seq);altered.gt+=100
        for config in CONFIGS:
            x=features(seq,config)
            self.assertEqual(x.shape,(99,expected[config]))
            np.testing.assert_array_equal(x,features(altered,config))

    def test_future_inputs_do_not_change_past_features(self):
        seq=sequence();altered=copy.deepcopy(seq);altered.odom[60:,0]+=100
        np.testing.assert_array_equal(features(seq,'relative32')[:59],
                                      features(altered,'relative32')[:59])
        x=np.arange(20)[:,None]
        self.assertEqual(causal_windows(x)[4,-1,0],4)
        self.assertLessEqual(causal_windows(x)[4].max(),4)

    def test_hard_fallback_removes_accumulated_drift(self):
        seq=sequence();prediction=np.zeros((99,3));prediction[:,0]=.01
        gates=np.ones(99);gates[50:]=0
        pose=corrected(seq,prediction,target='increment',gate=gates)
        self.assertGreater(pose[50,0]-seq.cart[50,0],.4)
        np.testing.assert_array_equal(pose[51:],seq.cart[51:])
        np.testing.assert_array_equal(corrected(seq,prediction,alpha=0,target='increment'),seq.cart)

    def test_metrics_analytical_motion(self):
        seq=sequence();scaled=seq.cart.copy();scaled[:,0]*=1.1
        report=metrics(seq,scaled)
        self.assertAlmostEqual(report['rpe']['1s']['translation_m']['rmse'],.1,places=5)
        self.assertEqual(metrics(seq,seq.cart)['ape_m']['rmse'],0)

    def test_uncertainty_loss_finite_at_extreme_variances(self):
        output=torch.zeros(4,6,requires_grad=True)
        with torch.no_grad():output[:,3:]=100
        value=loss(output,torch.ones(4,3),True)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(output.grad).all())

    def test_portable_causal_features_match_training(self):
        seq=sequence()
        seq.odom[:,0]+=.02*np.sin(seq.t)
        seq.cart[:,2]=.1*np.sin(seq.t)
        expected=features(seq,'relative32')
        for end in (2,3,5,20,100):
            start=max(0,end-6)
            actual=motion_feature(seq.t[start:end],seq.cart[start:end],seq.odom[start:end])
            np.testing.assert_allclose(actual,expected[end-2],atol=1e-6,rtol=1e-6)

    def test_tcn_uses_only_past_window(self):
        model=Regressor(6,'tcn').eval()
        x=torch.randn(3,5,6)
        self.assertEqual(tuple(model(x).shape),(3,3))
        activations=[]
        handle=model.encoder[1].register_forward_hook(lambda module,inputs,output:activations.append(output.detach().clone()))
        model(x)
        changed=x.clone();changed[:,3:]+=100
        model(changed);handle.remove()
        torch.testing.assert_close(activations[0][:,:,:3],activations[1][:,:,:3])

    def test_scoring_rigid_alignment_recovers_rotation_without_scale(self):
        truth=sequence().gt
        truth[:,1]=np.sin(truth[:,0])
        angle=.7;c,s=np.cos(angle),np.sin(angle)
        shifted=truth.copy();shifted[:,:2]=truth[:,:2]@np.array([[c,s],[-s,c]])+[3,-4]
        shifted[:,2]+=angle
        aligned=rigid_alignment(shifted,truth,np.ones(len(truth),dtype=bool))
        np.testing.assert_allclose(aligned,truth,atol=1e-10)
        scaled=shifted.copy();scaled[:,:2]*=1.1
        self.assertGreater(np.linalg.norm(rigid_alignment(scaled,truth,np.ones(len(truth),dtype=bool))[:,:2]-truth[:,:2]),.1)

    def test_online_missing_odometry_publishes_exact_original_pose(self):
        from collections import deque
        from tools.online_ros_benchmark import OnlineCorrection
        online=object.__new__(OnlineCorrection)
        online.last_time=0;online.history=deque([(0,None,None)],maxlen=6);online.last_output=np.ones(3)
        captured=[]
        online._publish=lambda stamp,pose,start,fallback:captured.append((stamp,pose.copy(),fallback))
        pose=np.eye(4);pose[:3,3]=[1,2,3]
        online.apply(.2,pose,None)
        self.assertEqual(len(captured),1)
        np.testing.assert_array_equal(captured[0][1],pose)
        self.assertTrue(captured[0][2]);self.assertEqual(len(online.history),0)

    def test_shared_map_scoring_preserves_correction_at_anchor(self):
        truth=sequence().gt
        raw=truth.copy();raw[:,:2]+=[3,-4]
        correction=raw.copy();correction[:,0]+=.1;correction[:,2]+=.02
        aligned=shared_anchor_alignment(correction,raw,truth,0)
        np.testing.assert_allclose(aligned[:,0]-truth[:,0],.1,atol=1e-12)
        np.testing.assert_allclose(aligned[:,2]-truth[:,2],.02,atol=1e-12)

    def test_portable_low_confidence_removes_previous_integrated_offset(self):
        from collections import deque
        model=object.__new__(PortableResidual)
        model.history=deque(maxlen=6);model.previous=None
        model.metadata={'alpha':1,'gate_parameters':{'gate_alpha':1,'sigma_threshold':1,'ood_threshold':100}}
        model.weights={'ys':np.ones(3),'ym':np.zeros(3),'clip':np.ones(3)}
        variance=[-8.]
        model.forward=lambda feature:(np.array([.01,0,0,*([variance[0]]*3)]),np.zeros(32))
        for i in range(40):
            raw=np.array([i*.1,0,0]);pose,status=model.step(i*.1,raw,raw)
        self.assertGreater(pose[0]-raw[0],.3)
        variance[0]=6
        raw=np.array([4.,0,0]);pose,status=model.step(4.,raw,raw)
        self.assertTrue(status['fallback']);np.testing.assert_array_equal(pose,raw)
        variance[0]=-8
        raw=np.array([4.1,0,0]);pose,status=model.step(4.1,raw,raw)
        self.assertAlmostEqual(pose[0]-raw[0],.01)

    def test_reference_timestamps_do_not_set_input_grid(self):
        seq=sequence()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def write(path,t,p):
                np.savetxt(path,np.column_stack([t,p[:,:2],np.zeros(len(t)),
                           np.zeros((len(t),2)),np.sin(p[:,2]/2),np.cos(p[:,2]/2)]))
            for name in ['cart','odom','gt']:write(root/(name+'.tum'),seq.t,seq.cart)
            row={'sequence_id':'s','scene_id':'r','trajectory_kind':'online',
                 **{name:str(root/(name+'.tum')) for name in ['cart','odom','gt']}}
            a=load_sequence(row)
            write(root/'gt.tum',seq.t[10:],seq.gt[10:])
            b=load_sequence(row)
            np.testing.assert_array_equal(a.t,b.t)
            np.testing.assert_array_equal(features(a,'relative32'),features(b,'relative32'))


if __name__=='__main__':unittest.main()
