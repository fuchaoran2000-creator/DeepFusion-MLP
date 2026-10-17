"""Differentiable body-frame SE2 rollout used only for supervised training."""
import torch


def rollout(increments):
    yaw=torch.cumsum(increments[...,2],dim=1)
    previous=yaw-increments[...,2]
    c,s=torch.cos(previous),torch.sin(previous)
    x=c*increments[...,0]-s*increments[...,1]
    y=s*increments[...,0]+c*increments[...,1]
    return torch.stack([torch.cumsum(x,dim=1),torch.cumsum(y,dim=1),yaw],dim=-1)


def path_loss(residual,cart_increments,reference_increments,scale,horizons=(10,30,50)):
    estimate=rollout(cart_increments+residual);reference=rollout(reference_increments)
    terms=[]
    for h in horizons:
        delta=estimate[:,h-1]-reference[:,h-1]
        yaw=2*torch.sin(delta[:,2]/2)
        error=torch.cat([delta[:,:2],yaw[:,None]],dim=-1)/(scale*(h**.5))
        terms.append((error**2).mean())
    return torch.stack(terms).mean()
