import json
import unittest
from pathlib import Path
import numpy as np
import torch

from causal_stream import extract
from mentor_benchmark import Sequence
from gate_stress import corrupt,gate_masks,first_streak,new_rejection_latency,evaluate


class GateStressTests(unittest.TestCase):
    def data(self):
        t=np.arange(120)*.1
        cart=np.column_stack([10+t,3+np.sin(t)*.1,np.zeros(len(t))])
        return {'t':t,'cart':cart,'odom':cart.copy(),'available':np.ones(len(t),bool),
                'hz':np.array(10.),'odom_source_t':t.copy(),'cart_source_t':t.copy()}

    def test_hidden_time_delay_is_causal_and_visible_stale_is_rejected(self):
        data=self.data()
        delayed,mask,start,end=corrupt(data,'odom_delay',{'lag_s':.2})
        np.testing.assert_array_equal(delayed['odom'][start:end],data['odom'][start-2:end-2])
        np.testing.assert_array_equal(delayed['odom_source_t'],data['odom_source_t'])
        np.testing.assert_array_equal(delayed['available'],data['available'])
        stale,_,_,_=corrupt(data,'odom_stale_stamp',{'source_timestamp_offset_s':-.2,'age_guard_s':.15})
        self.assertFalse(stale['available'][mask].any())
        np.testing.assert_array_equal(data['odom'],data['cart'])

    def test_cart_outage_holds_already_received_pose_for_two_seconds(self):
        data=self.data();altered,mask,start,end=corrupt(data,'cart_outage',{'duration_s':2.})
        self.assertEqual(end-start,20)
        np.testing.assert_array_equal(altered['cart'][mask],np.tile(data['cart'][start-1],(20,1)))
        self.assertFalse(altered['available'][mask].any())
        np.testing.assert_array_equal(altered['cart'][end:],data['cart'][end:])

    def test_sigma_addition_is_the_only_difference_from_rule_gate(self):
        ready=np.array([True,False,True,True]);z=np.array([[.2],[0],[100],[.1]],np.float32)
        output=np.zeros((4,6),np.float32);output[0,3:]=np.log(4);output[3,3:]=-8
        masks=gate_masks(ready,z,output,{'ood_threshold':1.,'sigma_threshold':1.})
        np.testing.assert_array_equal(masks['none'],[True,False,True,True])
        np.testing.assert_array_equal(masks['rule'],[True,False,False,True])
        np.testing.assert_array_equal(masks['uncertainty'],[False,False,False,True])

    def test_recovery_requires_three_acceptances_and_censors_missing_streak(self):
        t=np.arange(6)*.1
        self.assertIsNone(first_streak(np.array([False,False,False,False,True,True]),2,t))
        self.assertAlmostEqual(first_streak(np.array([False,False,False,True,True,True]),2,t),.1)
        affected=np.array([False,False,True,True,True,False])
        fallback=np.ones(6,bool)
        self.assertIsNone(new_rejection_latency(fallback,fallback,affected,t))
        self.assertAlmostEqual(new_rejection_latency(fallback,~affected,affected,t),0.)

    def test_fixed_gt_mask_same_alpha_and_exact_missing_frame_fallback(self):
        data=self.data();cart,odom,_,ready,_=extract(data['t'],data['cart'],data['odom'],data['available'],'full20')
        seq=Sequence('s','room',data['t'],cart,odom,cart.copy(),np.ones(120,bool),'online',{},data['available'],ready,'causal_grid_v1')
        state={'config':'full20','target':'pose','seed':11,'norm':{
            'xm':torch.zeros(20),'xs':torch.ones(20),'ym':torch.zeros(3),'ys':torch.ones(3),'clip':torch.ones(3)},
            'gate_parameters':{'gate_alpha':.65,'ood_threshold':1e8,'sigma_threshold':1e8}}
        def model(x):
            out=torch.zeros(len(x),6);out[:,0]=.1;return out
        protocol=json.loads(Path('config/causal_gate_stress_v1.json').read_text())
        protocol['corruptions']={'clean':{},'odom_outage':{'duration_s':2.}}
        captured=[]
        rows=evaluate(seq,data,state,model,protocol,lambda name,mode,work,pose,active:captured.append((name,work,pose,active)))
        self.assertTrue(all(r['alpha']==.65 for r in rows))
        self.assertTrue(all(r['scored_samples']==120 for r in rows))
        self.assertTrue(all(r['same_clean_gt_mask'] for r in rows))
        for name,work,pose,active in captured:
            if name=='odom_outage':
                np.testing.assert_array_equal(pose[40:60],work.cart[40:60])
                self.assertFalse(active[40:61].any())
                np.testing.assert_array_equal(work.valid,seq.valid)


if __name__=='__main__':unittest.main()
