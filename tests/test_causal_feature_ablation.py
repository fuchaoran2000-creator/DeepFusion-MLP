import unittest
import numpy as np
from mentor_benchmark import Sequence, GROUPS, features
from tools.train_causal_feature_ablation import subset


class CausalAblationTests(unittest.TestCase):
    def setUp(self):
        t=np.arange(9)*.1
        cart=np.column_stack([t,t*t,np.linspace(3.,3.4,9)])
        odom=np.column_stack([t*1.1,t*t*.9,np.linspace(3.1,3.6,9)])
        available=np.ones(9,dtype=bool);available[4]=False
        self.seq=Sequence('synthetic','synthetic',t,cart,odom,cart.copy(),available.copy(),
                          'online',{},available,stream_protocol='causal_grid_v1')

    def test_group_mask_retains_unmasked_features_and_budget(self):
        full=features(self.seq,'full20')
        for group,columns in GROUPS.items():
            masked=subset(self.seq,'mask_'+group)
            self.assertEqual(masked.shape,full.shape)
            expected=full.copy();expected[:,columns]=0
            np.testing.assert_array_equal(masked,expected)

    def test_compact_controls_preserve_shared_causal_features(self):
        full=features(self.seq,'full20');r=features(self.seq,'relative32')
        np.testing.assert_array_equal(subset(self.seq,'poses8'),full[:,:8])
        np.testing.assert_array_equal(subset(self.seq,'motion_difference12'),full[:,8:])
        np.testing.assert_array_equal(subset(self.seq,'relative6_pose'),r[:,:6])
        np.testing.assert_array_equal(subset(self.seq,'relative32_pose'),r)

    def test_missing_and_first_recovery_are_zero_for_every_representation(self):
        for config in ['full20','mask_increments','poses8','relative6_pose','relative32_pose']:
            x=subset(self.seq,config)
            np.testing.assert_array_equal(x[3:5],np.zeros_like(x[3:5]))
            self.assertTrue(np.isfinite(x).all())


if __name__=='__main__':unittest.main()
