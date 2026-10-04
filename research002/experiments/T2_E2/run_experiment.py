"""T2-E2: development sensitivity gate, frozen main fit, then one held-out evaluation."""
from __future__ import annotations
import argparse
import json
import platform
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
import legacy_t2e1 as legacy
from observer import fit_mlp,fit_ridge,predict
from study_common import *
from aggregate_results import aggregate


def now():return datetime.now(timezone.utc).isoformat()


def fit_panel(views,y,seed,step,phase,folder,arms):
    folder.mkdir(parents=True,exist_ok=False)
    offset=0 if phase=='development' else 10000000
    fitted={};details={}
    for arm in arms:
        if arm=='current_reference':continue
        if arm=='shuffled_mlp' and step==2:
            fitted[arm]=fitted['history_mlp']
            details[arm]={**details['history_mlp'],'alias_of':'history_mlp','fit_seconds':0.}
            continue
        x=views[arm_view(arm)]
        if arm.endswith('_mlp'):
            model,detail=fit_mlp(x[TRAIN],y[TRAIN],x[VAL],y[VAL],seed=seed,step=step,phase_offset=offset)
        else:model,detail=fit_ridge(x[TRAIN],y[TRAIN],x[VAL],y[VAL])
        fitted[arm]=model
        detail['extra_rff_projection_macs']=388*36*(step-1) if arm_view(arm)=='current_rff' else 0
        detail['history_float32_bytes_per_example']=4*36*(step-1) if arm_view(arm) in ['history','shuffled','mismatched'] else 0
        details[arm]=detail
        np.savez_compressed(folder/f'{arm}.npz',**model)
    reference=min(REFERENCE_ARMS,key=lambda arm:(details[arm]['validation_mse'],REFERENCE_ARMS.index(arm)))
    details['current_reference']={**details[reference],'alias_of':reference,'fit_seconds':0.}
    fitted['current_reference']=fitted[reference]
    dump(folder/'selection.json',details)
    return fitted,details


def predict_panel(fitted,details,views,arms):
    cols=[]
    for arm in arms:
        actual=details[arm].get('alias_of',arm)
        cols.append(predict(fitted[arm],views[arm_view(actual)]))
    return np.stack(cols,axis=-1)


def load_panel(folder,arms):
    details=json.loads((folder/'selection.json').read_text())
    models={}
    for arm in arms:
        actual=details[arm].get('alias_of',arm)
        with np.load(folder/f'{actual}.npz',allow_pickle=False) as z:models[arm]={k:z[k] for k in z.files}
    return models,details


def make_output(path):
    assert not path.exists(),'Use a new output path; overwriting a completed phase is forbidden'
    path.mkdir(parents=True)
    for name in ['data','predictions','models']:(path/name).mkdir()


def run_development(out):
    verify_inputs();make_output(out)
    started=time.time();seen=set();checks=[];oracles=[]
    study=legacy.load_config(ROOT/'frozen/model_protocol.yaml')
    for seed in SEEDS:
        tick=time.time()
        print(f'DEVELOPMENT seed={seed}: synthetic targets, real development representations',flush=True)
        tokens,labels,metadata,families=new_data(study,seed,'development',seen)
        seen.update(families);checks.append(metadata)
        np.savez_compressed(out/'data'/f'seed_{seed}.npz',tokens=tokens,labels=labels)
        full,cls,summaries,_=legacy.extract(legacy.load_model(study,seed),tokens)
        current,history=raw_views(full,cls,summaries,2)
        rng=np.random.default_rng(450000000+seed)
        z,epsilon=paired_gaussian(rng),paired_gaussian(rng)
        history[:,0]=z
        variances=current[TRAIN].var(axis=0)
        coordinate=int(np.argmax(variances))
        g=(current[:,coordinate]-current[TRAIN,coordinate].mean())/current[TRAIN,coordinate].std()
        oracle_current=1+np.sqrt(.5)*g
        transform=transformations(current,history,seed,2,'development')
        views=matrices(current,history,transform,seed,2,'development')
        predictions=[];targets=[]
        for ci,delta in enumerate([0.,.025,.1]):
            oracle_history=oracle_current+np.sqrt(delta)*z
            target=oracle_history+np.sqrt(.5-delta)*epsilon
            models,details=fit_panel(views,target,seed,2,'development',out/'models'/f'seed_{seed}_case_{ci}',DEV_ARMS)
            predictions.append(predict_panel(models,details,{k:x[TEST] for k,x in views.items()},DEV_ARMS))
            targets.append(target[TEST])
            e0=np.mean((target[TEST]-oracle_current[TEST])**2)
            e1=np.mean((target[TEST]-oracle_history[TEST])**2)
            oracles.append({'seed':seed,'delta_mse':delta,'population_relative_oracle_gain':2*delta,
                'current_teacher_coordinate':coordinate,'realized_current_oracle_mse':float(e0),
                'realized_history_oracle_mse':float(e1),'realized_relative_oracle_gain':float((e0-e1)/e0)})
            print(f'DEVELOPMENT seed={seed} oracle_gain={2*delta:.0%} fits complete',flush=True)
        np.savez_compressed(out/'predictions'/f'seed_{seed}.npz',target=np.stack(targets,axis=1),
            prediction=np.stack(predictions,axis=1),eligible=np.ones(3,dtype=bool))
        dump(out/'data_checks.json',checks);dump(out/'oracle_checks.json',oracles)
        print(f'DEVELOPMENT seed={seed} finished in {time.time()-tick:.1f}s',flush=True)
    panels,decisions=aggregate(out,development=True)
    go=not decisions[0]['screen_passed'] and decisions[-1]['screen_passed']
    result={'study_id':'T2-E2','phase':'development_calibration','completed_utc':now(),
        'elapsed_seconds':time.time()-started,'go_to_main':bool(go),'individual_mlp_training_runs':216,
        'ridge_paths':36,'synthetic_scenarios':3,'main_data_generated':False,'natural_history_evidence_added':False,
        'protocol_sha256':sha(ROOT/'STUDY_PROTOCOL.json')}
    dump(out/'development_result.json',result);verify_inputs()
    print('DEVELOPMENT COMPLETE',json.dumps(result),flush=True)
    for row in panels:
        if row['baseline'] in ['current_rff_mlp','current_reference']:print(json.dumps(row),flush=True)


def freeze_main(development,out):
    dev=json.loads((development/'development_result.json').read_text())
    assert dev['go_to_main'],'Development gate failed: main data must remain unopened'
    assert not out.exists()
    sources={p.name:sha(p) for p in ROOT.glob('*.py')}
    record={'study_id':'T2-E2','frozen_utc':now(),'development_result_sha256':sha(development/'development_result.json'),
        'protocol_sha256':sha(ROOT/'STUDY_PROTOCOL.json'),'sources':sources,
        'main_data_generated':False,'development_go_rule_passed':True,'scientific_protocol_changes':[]}
    assert not (ROOT/'MAIN_FREEZE.json').exists()
    dump(ROOT/'MAIN_FREEZE.json',record)


def verify_main_freeze():
    record=json.loads((ROOT/'MAIN_FREEZE.json').read_text())
    assert record['protocol_sha256']==sha(ROOT/'STUDY_PROTOCOL.json')
    for name,digest in record['sources'].items():assert sha(ROOT/name)==digest,name


def fit_main(development,out):
    verify_inputs();verify_main_freeze();make_output(out)
    started=time.time();seen=set();checks=[];diagnostics=[]
    for seed in SEEDS:
        with np.load(development/'data'/f'seed_{seed}.npz') as z:seen.update(legacy.family_keys(z['tokens']))
    study=legacy.load_config(ROOT/'frozen/model_protocol.yaml')
    for seed in SEEDS:
        tick=time.time();print(f'MAIN FIT seed={seed}: training/validation observations only',flush=True)
        tokens,labels,metadata,families=new_data(study,seed,'main',seen)
        seen.update(families);checks.append(metadata)
        np.savez_compressed(out/'data'/f'seed_{seed}.npz',tokens=tokens,labels=labels)
        # Only the first4,000 rows enter model inference during main fitting.
        full,cls,summaries,probs=legacy.extract(legacy.load_model(study,seed),tokens[:4000])
        target=natural_targets(probs,labels[:4000])
        for si,step in enumerate(STEPS):
            eligible=bool(target[TRAIN,si].mean()>=.02 and target[TRAIN,si].std(ddof=1)>=.02)
            diagnostics.append({'seed':seed,'step':step,'eligible':eligible,'train_target_mean':float(target[TRAIN,si].mean()),
                'train_target_sd':float(target[TRAIN,si].std(ddof=1))})
            current,history=raw_views(full,cls,summaries,step)
            transform=transformations(current,history,seed,step,'main')
            views=matrices(current,history,transform,seed,step,'main')
            folder=out/'models'/f'seed_{seed}_step_{step}'
            fit_panel(views,target[:,si],seed,step,'main',folder,ARMS)
            np.savez_compressed(folder/'feature_transform.npz',**transform)
            print(f'MAIN FIT seed={seed} step={step} completed; training eligibility={eligible}',flush=True)
        dump(out/'data_checks.json',checks);dump(out/'training_diagnostics.json',diagnostics)
        print(f'MAIN FIT seed={seed} finished in {time.time()-tick:.1f}s',flush=True)
    fit_files={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    dump(out/'FIT_MANIFEST.json',fit_files)
    dump(out/'fit_result.json',{'completed_utc':now(),'elapsed_seconds':time.time()-started,
        'individual_mlp_training_runs':456,'ridge_paths':72,'natural_test_predictions_computed':False,
        'natural_test_metrics_computed':False,'protocol_sha256':sha(ROOT/'STUDY_PROTOCOL.json')})
    verify_main_freeze();verify_inputs()
    print('MAIN FIT COMPLETE: all probe choices fixed; test outcomes still unopened',flush=True)


def evaluate_main(development,out):
    assert not (out/'study_result.json').exists(),'Completed evaluation cannot be overwritten'
    verify_main_freeze();verify_inputs()
    for name,digest in json.loads((out/'FIT_MANIFEST.json').read_text()).items():assert sha(out/name)==digest,name
    started=time.time();diagnostics=json.loads((out/'training_diagnostics.json').read_text());test_diagnostics=[]
    study=legacy.load_config(ROOT/'frozen/model_protocol.yaml')
    for seed in SEEDS:
        with np.load(out/'data'/f'seed_{seed}.npz') as z:tokens,labels=z['tokens'][TEST],z['labels'][TEST]
        full,cls,summaries,probs=legacy.extract(legacy.load_model(study,seed),tokens)
        target=natural_targets(probs,labels);predictions=[];eligible=[]
        for si,step in enumerate(STEPS):
            record=next(r for r in diagnostics if r['seed']==seed and r['step']==step)
            eligible.append(record['eligible'])
            current,history=raw_views(full,cls,summaries,step)
            folder=out/'models'/f'seed_{seed}_step_{step}'
            with np.load(folder/'feature_transform.npz') as z:transform={k:z[k] for k in z.files}
            views=matrices(current,history,transform,seed,step,'main',partition_start=2)
            fitted,details=load_panel(folder,ARMS)
            predictions.append(predict_panel(fitted,details,views,ARMS))
            test_diagnostics.append({**record,'test_target_mean':float(target[:,si].mean()),
                'test_target_sd':float(target[:,si].std(ddof=1)),
                'test_final_accuracy':float(np.mean(probs[:,-1].argmax(axis=1)==labels)),
                'test_flip_prevalence':float(np.mean(probs[:,step].argmax(axis=1)!=probs[:,-1].argmax(axis=1)))})
        np.savez_compressed(out/'predictions'/f'seed_{seed}.npz',target=target,prediction=np.stack(predictions,axis=1),eligible=np.array(eligible))
        print(f'MAIN EVALUATION seed={seed} complete',flush=True)
    dump(out/'test_diagnostics.json',test_diagnostics)
    panels,decisions=aggregate(out)
    for name,digest in json.loads((out/'FIT_MANIFEST.json').read_text()).items():assert sha(out/name)==digest,name
    result={'study_id':'T2-E2','status':'completed','completed_utc':now(),'decision':decisions[0],
        'evaluation_seconds':time.time()-started,'main_examples':36000,'development_examples':36000,
        'main_reported_cells':216,'main_mlp_training_runs':456,'main_ridge_paths':72,
        'base_model_training_updates':0,'reserved_model_seeds_opened':[],'confirmation_advancement':False,
        'protocol_sha256':sha(ROOT/'STUDY_PROTOCOL.json'),'main_freeze_sha256':sha(ROOT/'MAIN_FREEZE.json'),
        'fit_manifest_unchanged':True,'test_evaluated_after_all_model_selection':True,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}}
    dump(out/'study_result.json',result);verify_main_freeze();verify_inputs()
    print('MAIN STUDY COMPLETE',json.dumps(result),flush=True)
    for row in panels:
        if row['baseline'] in ['current_rff_mlp','current_reference']:print(json.dumps(row),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['development','freeze','fit','evaluate'])
    parser.add_argument('--development',type=Path,default=ROOT/'development')
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.phase=='development':run_development(args.development.resolve())
        elif args.phase=='freeze':freeze_main(args.development.resolve(),args.out.resolve())
        elif args.phase=='fit':fit_main(args.development.resolve(),args.out.resolve())
        else:evaluate_main(args.development.resolve(),args.out.resolve())
