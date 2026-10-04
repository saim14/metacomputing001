"""Calibration-only engineering benchmark; never accesses F1-A2 planning outcomes."""
from pathlib import Path
import hashlib,importlib.util,json,platform,time
import numpy as np
from threadpoolctl import threadpool_limits
from cheap_bias import CompactResidual,fixtures

ROOT=Path(__file__).resolve().parent;A2=ROOT.parent/'F1_A2'
def main():
    output=ROOT/'methods_pilot.json'
    if output.exists():raise RuntimeError('Preserve an existing pilot; choose a new output explicitly.')
    checks=fixtures()
    spec=importlib.util.spec_from_file_location('f1a2_methods',A2/'study.py')
    a2=importlib.util.module_from_spec(spec);spec.loader.exec_module(a2)
    path=A2/'run/calibration_shared.npz';x=np.load(path)['x']
    order=np.random.default_rng(73002).permutation(len(x));fit,test=order[:1536],order[1536:]
    assert len(np.intersect1d(fit,test))==0
    rows=[];sources={str(path.relative_to(ROOT.parent)):hashlib.sha256(path.read_bytes()).hexdigest()}
    for name in ['m3109_a100','m3119_a100','m3137_a100']:
        p=A2/f'run/calibration_{name}.npz';residual=np.load(p)['residual'];sources[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
        estimators={'local_linear':a2.Bank(x[fit],residual[fit]),'compact_quadratic':CompactResidual(x[fit],residual[fit])}
        q=x[test]
        for label,est in estimators.items():
            pred=est.correct(q[:,:2],q[:,2:]);timings=[]
            for _ in range(21):
                start=time.perf_counter();again=est.correct(q[:,:2],q[:,2:]);timings.append(time.perf_counter()-start)
                assert np.array_equal(pred,again)
            rows.append({'condition':name,'estimator':label,'heldout_residual_rmse':float(np.sqrt(np.mean((pred-residual[test])**2))),
                'zero_correction_rmse':float(np.sqrt(np.mean(residual[test]**2))),
                'seconds_per_512_queries_median':float(np.median(timings[1:])),'timings_seconds':timings[1:]})
    result={'status':'engineering_methods_pilot_only','main_study_run':False,'python':platform.python_version(),'numpy':np.__version__,
       'fit_rows':1536,'heldout_rows':512,'split_seed':73002,'fixed_ridge':.001,'fixtures':checks,'source_sha256':sources,
       'rows':rows,'limits':['Inherited calibration bank; not independent confirmation','No planning-value labels or controller evaluation','Query timing excludes rollout, model inference and feature readout'],
       'full_candidate_model_calls':320,'proposed_8_candidate_model_calls':80}
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'fixtures':checks,'rows':[{k:v for k,v in r.items() if k!='timings_seconds'} for r in rows]},indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
