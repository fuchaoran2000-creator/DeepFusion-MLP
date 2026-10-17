"""Differentiable runtime-policy rollout; training labels stay outside runtime."""
import math
from typing import Optional,List
import torch
from trajectory_objective import rollout


@torch.jit.script
def policy_rollout(residual:torch.Tensor,ci:torch.Tensor,accepted:torch.Tensor,
                   alpha:float=1.,tau_s:Optional[float]=None,dt:float=.1):
    """Windows start at raw pose zero; rejection resets all accumulated offsets."""
    raw=rollout(ci)
    if alpha==0.:return raw
    current=torch.zeros_like(raw[:,0])
    result=torch.jit.annotate(List[torch.Tensor], [])
    retain=1. if tau_s is None else math.exp(-dt/tau_s)
    for i in range(ci.shape[1]):
        inc=ci[:,i]+alpha*residual[:,i]
        c,s=torch.cos(current[:,2]),torch.sin(current[:,2])
        candidate=torch.stack([current[:,0]+c*inc[:,0]-s*inc[:,1],
            current[:,1]+s*inc[:,0]+c*inc[:,1],current[:,2]+inc[:,2]],dim=1)
        yaw=torch.remainder(candidate[:,2]-raw[:,i,2]+math.pi,2*math.pi)-math.pi
        offset=torch.cat([candidate[:,:2]-raw[:,i,:2],yaw[:,None]],dim=1)
        candidate=raw[:,i]+retain*offset
        candidate=torch.cat([candidate[:,:2],(torch.remainder(candidate[:,2]+math.pi,2*math.pi)-math.pi)[:,None]],dim=1)
        current=torch.where(accepted[:,i,None],candidate,raw[:,i])
        result.append(current)
    return torch.stack(result,dim=1)


def gated_path_loss(residual,ci,gi,accepted,scale,alpha=1.,tau_s=None,horizons=(10,30,50)):
    if tau_s is not None and (not math.isfinite(tau_s) or tau_s<=0):raise ValueError('Positive finite tau required')
    estimate=policy_rollout(residual,ci,accepted,alpha,tau_s)
    reference=rollout(gi);terms=[]
    for h in horizons:
        delta=estimate[:,h-1]-reference[:,h-1]
        error=torch.cat([delta[:,:2],(2*torch.sin(delta[:,2]/2))[:,None]],dim=1)/(scale*math.sqrt(h))
        terms.append((error**2).mean())
    return torch.stack(terms).mean()
