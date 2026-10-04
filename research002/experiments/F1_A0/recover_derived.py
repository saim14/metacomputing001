"""Preserve and reconstruct an incomplete derived NPZ; never fit a model."""
import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits
from run_f1 import ROOT, World, plan


def main():
    output=ROOT/'run'
    original=output/'planning_3137.npz'
    if zipfile.is_zipfile(original):
        raise RuntimeError('The recorded file is already complete; recovery is unnecessary.')
    old=original.read_bytes()
    immutable={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.npz') if p.name!=original.name}
    immutable['result.json']=hashlib.sha256((output/'result.json').read_bytes()).hexdigest()
    recovery=output/'recovery'
    recovery.mkdir(exist_ok=False)
    (recovery/'planning_3137.npz.truncated').write_bytes(old)
    sc=np.load(output/'scenarios.npz')['scenarios']
    with np.load(output/'world_3137.npz') as w:
        model=World(dict(w))
    data,_=plan(model,sc,recovery,3137)
    rebuilt=recovery/'planning_3137.npz'
    with rebuilt.open('rb') as f:os.fsync(f.fileno())
    with zipfile.ZipFile(rebuilt) as z:assert z.testzip() is None
    new=rebuilt.read_bytes()
    assert new[:len(old)]==old, 'Reconstruction must preserve every existing byte.'
    ev=np.load(output/'evaluation.npz')
    ids=ev['scenario_ids']
    assert np.array_equal(data['target'][ids],ev['target'][2])
    assert np.array_equal(data['base'][ids],ev['base'][2])
    assert np.array_equal(data['final'][ids],ev['final'][2])
    os.replace(rebuilt,original)
    for name,digest in immutable.items():
        assert hashlib.sha256((output/name).read_bytes()).hexdigest()==digest
    record={'derived_record':'planning_3137.npz','old_bytes':len(old),'new_bytes':len(new),
            'old_sha256':hashlib.sha256(old).hexdigest(),'new_sha256':hashlib.sha256(new).hexdigest(),
            'entire_original_is_exact_prefix':True,'evaluation_outcomes_exactly_preserved':True,
            'other_records_unchanged':True,'additional_model_fits':0,'additional_observer_fits':0,
            'method':'Replay identical six-round planner from frozen weights, scenarios and fixed noise; no new data or selection.'}
    (recovery/'recovery.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
