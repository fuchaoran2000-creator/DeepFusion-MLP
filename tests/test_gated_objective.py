import unittest
import numpy as np
import torch
from causal_stream import increments,wrap
from trajectory_quality import QualityResidualStream
from trajectory_objective import rollout,path_loss
from gated_objective import policy_rollout,gated_path_loss


class GatedObjectiveTests(unittest.TestCase):
    def test_runtime_exact_policy_with_yaw_resets_and_memory(self):
        t=np.arange(51)*.1
        ci=np.tile([.12,.01,.16],(50,1));cart=np.r_[np.zeros((1,3)),rollout(torch.tensor(ci[None])).numpy()[0]]
        residual=np.tile([.013,-.004,.009],(50,1))
        # Include confidence rejection and an input outage with first recovery.
        missing=np.zeros(51,bool);missing[18:21]=True
        q=np.ones((51,12));q[missing,11]=0
        accepted=np.ones(50,bool);accepted[7:10]=False;accepted[17:21]=False
        for tau in [None,5.,30.]:
            norm=dict(xm=np.zeros(44),xs=np.ones(44),ym=np.zeros(3),ys=np.ones(3),clip=np.ones(3))
            gate=dict(gate_alpha=.65,sigma_threshold=2.,ood_threshold=1e9)
            calls=[0]
            def forward(z):
                i=calls[0];calls[0]+=1
                # Match rejection by global stamp supplied through q's first value.
                global_i=int(round(float(z[32])/.1))-1
                logvar=6. if 7<=global_i<10 else 0.
                return np.r_[residual[0],np.full(3,logvar)]
            engine=QualityResidualStream('quality44',norm,forward,.65,gate,tau)
            out=[];mask=[]
            for stamp,c,quality in zip(t,cart,q):
                quality=quality.copy();quality[0]=stamp
                p,s=engine.step(stamp,c,c,quality,True);out.append(p);mask.append(not s['fallback'])
            np.testing.assert_array_equal(mask[1:],accepted)
            actual=policy_rollout(torch.tensor(residual[None]),torch.tensor(ci[None]),torch.tensor(accepted[None]),.65,tau).numpy()[0]
            delta=actual-np.array(out)[1:];delta[:,2]=wrap(delta[:,2])
            np.testing.assert_allclose(delta,0,atol=1e-12)
            np.testing.assert_array_equal(actual[~accepted,:2],cart[1:][~accepted,:2])

    def test_rejection_cuts_gradient_history(self):
        ci=torch.zeros(1,10,3,dtype=torch.double)
        residual=torch.full_like(ci,.001,requires_grad=True)
        mask=torch.ones(1,10,dtype=torch.bool);mask[:,5]=False
        estimate=policy_rollout(residual,ci,mask)
        estimate[:,-1,:2].sum().backward()
        np.testing.assert_array_equal(residual.grad[:,:6].numpy(),0.)
        self.assertGreater(float(residual.grad[:,6:].abs().sum()),0.)

    def test_alpha_zero_and_all_rejected_are_raw(self):
        ci=torch.randn(2,50,3,dtype=torch.double)*.1;r=torch.randn_like(ci)*.02
        for mask,alpha in [(torch.ones(2,50,dtype=torch.bool),0.),(torch.zeros(2,50,dtype=torch.bool),1.)]:
            torch.testing.assert_close(policy_rollout(r,ci,mask,alpha),rollout(ci),atol=0,rtol=0)

    def test_continuous_policy_and_oracle_gradients(self):
        ci=torch.randn(2,50,3,dtype=torch.double)*.01;r=torch.randn_like(ci)*.001
        gi=ci+r;mask=torch.ones(2,50,dtype=torch.bool)
        loss=gated_path_loss(r,ci,gi,mask,torch.ones(3,dtype=torch.double))
        self.assertLess(float(loss),1e-25)
        bad=r.clone().requires_grad_()
        loss=gated_path_loss(bad+.001,ci,gi,mask,torch.ones(3,dtype=torch.double),.7,5.)
        loss.backward();self.assertTrue(torch.isfinite(bad.grad).all());self.assertGreater(float(bad.grad.abs().sum()),0.)
        a=gated_path_loss(r+.001,ci,gi,mask,torch.ones(3,dtype=torch.double))
        b=path_loss(r+.001,ci,gi,torch.ones(3,dtype=torch.double))
        torch.testing.assert_close(a,b,atol=1e-15,rtol=1e-12)


if __name__=='__main__':unittest.main()
