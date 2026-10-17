import unittest
from types import SimpleNamespace
import numpy as np
from tools.train_scene_rotation import make_folds,balanced_quantile,macro_score


class SceneRotationTests(unittest.TestCase):
    def test_each_scene_rotates_without_scene_or_sequence_leakage(self):
        rows=[dict(sequence_id=s+'1',scene_id=s,cart='/fake/'+s+'.tum',sampled_stream='/fake/'+s+'.npz',
                   stream_protocol='causal_grid_v1') for s in ['home','office','cafe','corridor','nclt']]
        folds=make_folds(dict(sequences=rows))
        self.assertEqual(len(folds),12)
        for f in folds:
            splits=[set(f[k]) for k in ['train','validation','test']]
            self.assertEqual([len(s) for s in splits],[2,1,1])
            self.assertEqual(set.union(*splits),{'home1','office1','cafe1','corridor1'})
            self.assertTrue(all(not splits[i]&splits[j] for i in range(3) for j in range(i)))
            self.assertEqual(f['external_test'],['nclt-2013-01-10'])
        for scene in ['home','office','cafe','corridor']:
            self.assertEqual(sum(f['test']==[scene+'1'] for f in folds),3)
            self.assertEqual(sum(f['validation']==[scene+'1'] for f in folds),3)

    def test_calibration_weights_do_not_favor_longer_sequences(self):
        groups=[[np.array([1.,2.]),np.array([3.,4.])],[np.array([10.,20.])]]
        repeated=[[np.repeat(groups[0][0],100),groups[0][1]],groups[1]]
        for q in [.1,.5,.9,.99]:self.assertEqual(balanced_quantile(groups,q),balanced_quantile(repeated,q))

    def test_pooled_relative_score_balances_scenes(self):
        data=[];poses=[]
        for scene,raw,error in [('cafe',.2,.1),('corridor',1.,1.)]:
            gt=np.zeros((30,3));cart=gt.copy();cart[:,0]=raw
            seq=SimpleNamespace(scene=scene,cart=cart,gt=gt,valid=np.ones(30,bool))
            data.append(dict(seq=seq));pose=gt.copy();pose[:,0]=error;poses.append(pose)
        self.assertAlmostEqual(macro_score(data,poses),.55)
        self.assertAlmostEqual(macro_score(data,poses,relative=True),.75)


if __name__=='__main__':unittest.main()
