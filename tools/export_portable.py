#!/usr/bin/python3
import argparse
import json
import sys
import shutil
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from residual_models import Regressor
from portable_residual import PortableResidual

parser=argparse.ArgumentParser();parser.add_argument('checkpoint',type=Path)
parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
if checkpoint['kind']!='mlp' or checkpoint['config']!='relative32' or checkpoint['target']!='increment':
    raise ValueError('Only relative32 increment MLP export supported')
args.output.mkdir(parents=True,exist_ok=True)
arrays={k:v.numpy() for k,v in checkpoint['norm'].items()}
for index,layer in enumerate([0,2,4,6]):
    arrays[f'w{index}']=checkpoint['model_state_dict'][f'net.{layer}.weight'].numpy()
    arrays[f'b{index}']=checkpoint['model_state_dict'][f'net.{layer}.bias'].numpy()
np.savez_compressed(args.output/'portable_weights.npz',**arrays)
metadata={'config':checkpoint['config'],'target':checkpoint['target'],'alpha':checkpoint['alpha'],
          'gate_parameters':checkpoint['gate_parameters'],'source_checkpoint':str(args.checkpoint.resolve()),
          'runtime_dependencies':['numpy'],'hardware_validation':'host CPU only; no edge device measurement',
          'input_contract':'two poses of the same robot reference point; each in a consistent estimator frame; timestamps in seconds'}
if checkpoint.get('stream_protocol'):
    metadata.update(stream_protocol=checkpoint['stream_protocol'],grid_hz=checkpoint['grid_hz'],
                    input_contract='already sampled causal grid pairs, same GridSampler as ROS; available=False resets history; preserve origins')
    metadata['runtime_source_files']=['portable_residual.py','causal_stream.py']
    if checkpoint.get('integration_tau_s') is not None:
        metadata['integration_tau_s']=checkpoint['integration_tau_s']
    for name in metadata['runtime_source_files']:
        shutil.copyfile(Path(__file__).resolve().parents[1]/name,args.output/name)
(args.output/'portable.json').write_text(json.dumps(metadata,indent=2)+'\n')
portable=PortableResidual(args.output)
model=Regressor(32,'mlp',checkpoint['uncertainty']);model.load_state_dict(checkpoint['model_state_dict']);model.eval()
rng=np.random.default_rng(91);errors=[]
for _ in range(100):
    feature=arrays['xm']+rng.normal(size=32).astype(np.float32)*arrays['xs']
    actual,_=portable.forward(feature)
    normalized=((feature-arrays['xm'])/arrays['xs']).astype(np.float32)
    with torch.inference_mode():expected=model(torch.from_numpy(normalized[None])).numpy()[0]
    np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
    errors.append(float(np.abs(actual-expected).max()))
metadata['export_verification']={'random_cases':100,'max_absolute_output_error':max(errors)}
(args.output/'portable.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps(metadata,indent=2),flush=True)
