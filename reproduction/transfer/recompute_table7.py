#!/usr/bin/env python3
"""Recalculate displayed archived transfer means with Python's standard library."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parent
ev=json.loads((root/'transfer_evidence.json').read_text())
paths=json.loads((root/'transfer_source_map.json').read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for source,value in ev['sources'].items():assert sha(root/paths[source])==value,source
assert sha(root/paths[ev['checkpoint']])==ev['checkpoint_sha256']
proof=[]
for row in ev['rows']:
    records=[r for r in csv.DictReader((root/paths[row['source']]).open())
             if r['sequence'] in row['sequences'] and r['config']=='full20' and r['seed']=='11'
             and r['anomaly']=='clean' and r['gate_mode']=='uncertainty']
    assert len(records)==len(row['sequences'])
    raw=sum(float(r['raw_clean_ape_m']) for r in records)/len(records)
    corrected=sum(float(r['ape_rmse_m']) for r in records)/len(records)
    gain=100*(raw-corrected)/raw
    assert raw==row['raw_ape_m'] and corrected==row['corrected_ape_m'] and gain==row['ape_gain_percent']
    proof.append(dict(scene=row['scene'],raw_ape_m=raw,corrected_ape_m=corrected,gain_percent=gain))
print(json.dumps(dict(passed=True,rows=proof,scope='Archived score aggregation and source hashes only; no new inference, fitting or SLAM execution.'),indent=2))
