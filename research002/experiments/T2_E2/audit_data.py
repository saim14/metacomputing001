"""Repeat the independent cross-phase family and prior-exposure audit."""
import argparse
import hashlib
import json
import numpy as np
import legacy_t2e1 as old
from study_common import ROOT,SEEDS,dump,old_exclusions


def audit(development,main):
    study=old.load_config(ROOT/'frozen/model_protocol.yaml')
    seen=set();rows=[]
    for phase in [development,main]:
        checks={r['seed']:r for r in json.loads((phase/'data_checks.json').read_text())}
        for seed in SEEDS:
            with np.load(phase/'data'/f'seed_{seed}.npz',allow_pickle=False) as z:x,y=z['tokens'],z['labels']
            keys=set(old.family_keys(x))
            assert len(keys)==3000 and not keys.intersection(seen)
            assert not keys.intersection(old_exclusions(study,seed))
            assert np.all(y[::2]+y[1::2]==1)
            assert hashlib.sha256(x.tobytes()).hexdigest()==checks[seed]['tokens_sha256']
            assert hashlib.sha256(y.tobytes()).hexdigest()==checks[seed]['labels_sha256']
            seen.update(keys);rows.append({'phase':phase.name,'seed':seed,'unique_families':len(keys),'overlap':0})
    return {'status':'passed','unique_families':len(seen),'examples':len(seen)*2,'prior_exposure_overlap':0,
        'cross_seed_cross_phase_overlap':0,'array_hashes_match':True,'checks':rows}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development',default='development')
    parser.add_argument('--main',default='results')
    args=parser.parse_args()
    result=audit(ROOT/args.development,ROOT/args.main)
    dump(ROOT/'cross_phase_data_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
