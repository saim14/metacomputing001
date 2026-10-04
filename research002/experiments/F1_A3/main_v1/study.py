"""F1-A3 main-v1. Fixed-budget, resumable comparison; no outcome-driven search."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, importlib.util, json, platform, time
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('a2_dependency',ROOT/'deps/study.py')
a2=importlib.util.module_from_spec(spec); spec.loader.exec_module(a2)
p=a2.prev; old=p.old
CFG=json.loads((ROOT/'config.json').read_text())
CONDITIONS=a2.CONDITIONS
REFS=a2.REFS
ARMS=REFS+['b32','b8','mb8','b1','xb8']
wj=p.write_json; wn=p.write_npz

def fingerprint():
    paths=[ROOT/n for n in ['study.py','config.json','F1_A3_Protocol.md']]
    paths+=sorted((ROOT/'deps').rglob('*'))
    return {str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in paths if f.is_file() and '__pycache__' not in f.parts}

def loadbank(condition):
    x=np.load(ROOT/'deps/run/calibration_shared.npz')['x']
    r=np.load(ROOT/f"deps/run/calibration_{condition['name']}.npz")['residual']
    return a2.Bank(x,r)

def subset(q):
    ranks=np.rint(np.linspace(0,q.shape[1]-1,CFG['subset_size'])).astype(int)
    return np.argsort(q,axis=1,kind='stable')[:,ranks]

def summary8(q,t):
    b=t-q; n,k=q.shape; row=np.arange(n); order=np.argsort(q,axis=1,kind='stable')
    chosen=order[:,0]; incumbent=b[row,chosen]; tc=t[row,chosen]
    qc=q-q.mean(1,keepdims=True); bc=b-b.mean(1,keepdims=True); cc=t-t.mean(1,keepdims=True)
    pair=np.triu_indices(k,1); e=CFG['subset_elites']
    discord=((q[:,pair[0]]-q[:,pair[1]])*(t[:,pair[0]]-t[:,pair[1]])<0).mean(1)
    return np.stack([incumbent,b.mean(1),b.std(1),b.min(1),b.max(1),
        np.take_along_axis(b,order[:,:e],1).mean(1)-incumbent,
        np.take_along_axis(b,order[:,e:],1).mean(1)-incumbent,
        (qc*bc).mean(1)/(np.mean(qc**2,1)+1e-12),
        (qc*cc).mean(1)/np.maximum(np.sqrt(np.mean(qc**2,1)*np.mean(cc**2,1)),1e-12),
        discord,tc-t.min(1),(t<tc[:,None]-.01).mean(1)],1).astype(np.float32)

def scenes():
    rng=np.random.default_rng(CFG['scene_seed']); rows=[]
    while len(rows)<CFG['n_scenarios']:
        start,goal=rng.uniform(-1.25,1.25,(2,2))
        if np.linalg.norm(start-goal)<.75:continue
        center=(start+goal)/2+rng.normal(0,.35,2); radius=rng.uniform(.15,.35)
        if min(np.linalg.norm(start-center),np.linalg.norm(goal-center))<=radius+.08:continue
        rows.append(np.r_[start,goal,center,radius])
    return np.array(rows)

def generate(model,bank,sc,noise):
    # Frozen A2 generation computes one common trajectory, full corrected costs,
    # and simulator diagnostics. Only prefix fields enter observer matrices.
    data,seconds=a2.generate(model,bank,sc,noise)
    ids=subset(data['model_cost']); q=np.take_along_axis(data['model_cost'],ids,1)
    t=np.take_along_axis(data['corrected_candidate_cost'],ids,1)
    data.update(b32=data.pop('b'),b8=summary8(q,t),subset_ids=ids)
    return data,seconds

def matrices(data,fit,splits,ci):
    c,h,g,r,raw,b32,b8=[data[k].astype(float) for k in ['c','h','g','r','raw','b32','b8']]
    cr=np.c_[c,r]; x=np.c_[c,h.reshape(len(c),-1),g,r]
    _,cm,cs=p.standardize(c,fit); _,rm,rs=p.standardize(cr,fit)
    mb,source=p.deranged_rows(b8,splits,np.random.default_rng(CFG['mismatch_seed']+ci))
    out={'c':c,'cr':cr,'ce':np.c_[c,a2.expansion(c,cm,cs)],
         'cre':np.c_[cr,a2.expansion(cr,rm,rs)],'craw':np.c_[c,raw],
         'crraw':np.c_[cr,raw],'x':x,'b32':np.c_[c,b32],'b8':np.c_[c,b8],
         'mb8':np.c_[c,mb],'b1':np.c_[c,b8[:,0]],'xb8':np.c_[x,b8]}
    return {k:v.astype(np.float32).astype(float) for k,v in out.items()},dict(cm=cm,cs=cs,rm=rm,rs=rs),source

def extra(arm):return {'b32':320,'b8':80,'mb8':80,'xb8':80,'b1':10}.get(arm,0)

def controller(pred,y,base,arm,charge=None):
    charge=CFG['charge'] if charge is None else charge
    go=pred>charge; loss=base-y*go
    return loss+charge*(go.astype(float)+extra(arm)/960),loss,go

def interval(v,boot):
    v=v.mean(0) if v.ndim==2 else v
    draws=v[boot].mean(1)
    return {'mean':float(v.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
            'ci_adjusted':np.quantile(draws,[.05/6,1-.05/6]).tolist()}

def compare(ref,candidate,y,boot):
    er=(ref-y)**2; ec=(candidate-y)**2
    return dict(interval(er-ec,boot),reference_mse=float(er.mean()),candidate_mse=float(ec.mean()),
                relative_reduction=float(1-ec.mean()/er.mean()))

def analyze(pred,y,base,refs,policyrefs):
    boot=np.random.default_rng(CFG['bootstrap_seed']).integers(0,y.shape[1],(CFG['bootstrap'],y.shape[1]))
    v={a:pred[:,i] for i,a in enumerate(ARMS)}; sl=slice(1,None)
    ref=np.array([v[a][i] for i,a in enumerate(refs)])
    pr=np.array([v[a][i] for i,a in enumerate(policyrefs)])
    signal=compare(ref[sl],v['b8'][sl],y[sl],boot)
    e8=(v['b8'][sl]-y[sl])**2; e32=(v['b32'][sl]-y[sl])**2
    diff=(e8-e32).mean(0); denom=e32.mean(0)
    ratios=diff[boot].mean(1)/denom[boot].mean(1)
    retention={'relative_mse_excess':float(diff.mean()/denom.mean()),
        'ci95':np.quantile(ratios,[.025,.975]).tolist(),
        'ci_adjusted':np.quantile(ratios,[.05/6,1-.05/6]).tolist(),'margin':CFG['noninferiority_margin']}
    cj,_,_=controller(v['b8'],y,base,'b8'); rj,_,_=controller(pr,y,base,'c')
    policy=dict(interval((rj-cj)[sl],boot),reference_objective=float(rj[sl].mean()),
        candidate_objective=float(cj[sl].mean()),relative_reduction=float(1-cj[sl].mean()/rj[sl].mean()))
    descriptive={k:compare(v[a][sl],v[b][sl],y[sl],boot) for k,a,b in [
        ('paired_subset','mb8','b8'),('beyond_incumbent','b1','b8'),('history_geometry','b8','xb8')]}
    descriptive['full_vs_reference']=compare(ref[sl],v['b32'][sl],y[sl],boot)
    descriptive['subset_vs_full']=compare(v['b32'][sl],v['b8'][sl],y[sl],boot)
    fullj,_,_=controller(v['b32'],y,base,'b32')
    descriptive['charged_subset_vs_full']=interval((fullj-cj)[sl],boot)
    per=[]
    for i,condition in enumerate(CONDITIONS):
        cs={}
        for arm in ['b8','b32','b1','xb8',policyrefs[i]]:
            j,l,go=controller(v[arm][i],y[i],base[i],arm); optimal=y[i]>CFG['charge']
            falsego=go&~optimal; falsestop=~go&optimal
            near=np.abs(v[arm][i]-CFG['charge'])<=.01
            cs[arm]={'charged_cost':float(j.mean()),'true_cost':float(l.mean()),'continue_fraction':float(go.mean()),
                'harmful_continuation_fraction':float((go&(y[i]<-.01)).mean()),
                'false_continue_regret':float(np.where(falsego,CFG['charge']-y[i],0).mean()),
                'false_stop_regret':float(np.where(falsestop,y[i]-CFG['charge'],0).mean()),
                'feature_charge':CFG['charge']*extra(arm)/960,
                'near_threshold_count':int(near.sum()),
                'near_threshold_wrong_fraction':float((go[near]!=optimal[near]).mean()) if near.any() else None,
                'mean_total_model_transitions':float(960+extra(arm)+960*go.mean())}
        per.append(dict(condition,reference=refs[i],controller_reference=policyrefs[i],
            harmful_fraction=float((y[i]<-.01).mean()),value_mean=float(y[i].mean()),
            mse={a:float(np.mean((v[a][i]-y[i])**2)) for a in ARMS},controllers=cs,
            always_stop=float(base[i].mean()),always_continue=float((base[i]-y[i]+CFG['charge']).mean()),
            hindsight_oracle=float((base[i]-np.maximum(y[i]-CFG['charge'],0)).mean())))
    verdicts={'useful_signal':signal['relative_reduction']>=CFG['prediction_gain'] and signal['ci_adjusted'][0]>0,
        'accuracy_retained':retention['ci_adjusted'][1]<CFG['noninferiority_margin'],
        'charged_controller':policy['relative_reduction']>=CFG['controller_gain'] and policy['ci_adjusted'][0]>0}
    return {'headline':dict(useful_signal=signal,accuracy_retained=retention,charged_controller=policy),
        'primary_verdicts':verdicts,'descriptive':descriptive,'per_condition':per,
        'pooled_mse':{a:float(np.mean((v[a][sl]-y[sl])**2)) for a in ARMS}}

def online_input(arm,sc,mean,sd,inc,paths,bp,values,best,actions,history,bank,model,expander):
    if arm in REFS:
        return a2.online_input(arm,sc,mean,sd,inc,paths,bp,values,best,actions,history,bank,model,expander)
    c=p.current(sc,mean,sd,inc,values,best).astype(float)
    ids=np.tile(np.arange(32),(len(sc),1)) if arm=='b32' else subset(values)
    suba=actions[np.arange(len(sc))[:,None],ids]; q=np.take_along_axis(values,ids,1)
    t=old.cost(old.rollout(a2.Corrected(model,bank),sc[:,:2],suba),suba,sc)
    b=a2.bias_summary(q,t) if arm=='b32' else summary8(q,t)
    return np.c_[c,b].astype(np.float32).astype(float)

class CountModel:
    def __init__(self,model):self.model=model; self.calls=0
    def __call__(self,x,a):
        self.calls+=int(np.prod(x.shape[:-1])); return self.model(x,a)

def online_case(model,bank,sc,noise,mode,predictor,expander):
    counted=CountModel(model); mean,sd,inc=p.init_planner(sc); history=[]; go=mode=='always_continue'
    for k in range(6):
        mean,sd,inc,paths,bp,values,best,actions=a2.step(counted,sc,mean,sd,inc,noise[k])
        if mode=='x' and k<2:history.append(p.current(sc,mean,sd,inc,values,best))
        if k==2:
            if mode=='always_stop':break
            if mode!='always_continue':
                x=online_input(mode,sc,mean,sd,inc,paths,bp,values,best,actions,history,bank,counted,expander)
                go=bool(predictor(x)[0]>CFG['charge'])
                if not go:break
    assert counted.calls==960+960*go+extra(mode)
    return inc,go,counted.calls

def benchmark(out,sc,noise,test,pred,policyrefs):
    rows=[]; total_mismatch=0; maxerr=0.
    for ci,condition in enumerate(CONDITIONS):
        name=condition['name']; path=out/f'runtime_{name}.json'
        if path.exists():
            row=json.loads(path.read_text()); rows.append(row)
            total_mismatch+=row['decision_mismatches']; maxerr=max(maxerr,row['plan_max_abs_error']); continue
        model=p.Model(condition); bank=loadbank(condition); ex=dict(np.load(out/f'expanders_{name}.npz'))
        d=np.load(out/f'data_{name}.npz'); ref=policyrefs[ci]
        modes=['b8','b32',ref,'always_stop','always_continue']
        predictors={a:p.Predictor(out/f'observer_{name}_{a}.npz') for a in modes[:3]}
        ids=test[:CFG['runtime_cases']]; timings=np.zeros((5,CFG['runtime_repeats'],len(ids)))
        counts=np.zeros_like(timings,dtype=int); mismatch=0; err=0.
        for mode in modes:online_case(model,bank,sc[ids[:1]],noise[:,ids[:1]],mode,predictors.get(mode),ex)
        rng=np.random.default_rng(CFG['runtime_seed']+ci)
        for rep in range(CFG['runtime_repeats']):
            for pos,sid in enumerate(ids):
                for mi in rng.permutation(5):
                    mode=modes[mi]; started=time.perf_counter()
                    plan,go,calls=online_case(model,bank,sc[sid:sid+1],noise[:,sid:sid+1],mode,predictors.get(mode),ex)
                    timings[mi,rep,pos]=(time.perf_counter()-started)*1000; counts[mi,rep,pos]=calls
                    expected=mode=='always_continue' if mode.startswith('always_') else bool(pred[ci,ARMS.index(mode),pos]>CFG['charge'])
                    mismatch+=int(go!=expected)
                    err=max(err,float(np.max(np.abs(plan-d['u6' if expected else 'u3'][sid:sid+1]))))
        means=np.median(timings,axis=1).mean(1)
        row={'condition':name,'modes':modes,'cases':len(ids),'repeats':CFG['runtime_repeats'],
            'ms_per_case':dict(zip(modes,map(float,means))),'decision_mismatches':mismatch,'plan_max_abs_error':err}
        wn(out/f'runtime_{name}.npz',milliseconds=timings,model_transitions=counts,scene_ids=ids)
        wj(path,row); rows.append(row); total_mismatch+=mismatch; maxerr=max(maxerr,err)
        print(json.dumps({'stage':'runtime',**row}),flush=True)
    ratios=[r['ms_per_case']['b8']/r['ms_per_case']['b32'] for r in rows[1:]]
    vsref=[r['ms_per_case']['b8']/r['ms_per_case'][policyrefs[i]] for i,r in enumerate(rows) if i>0]
    boot=np.random.default_rng(CFG['runtime_seed']+99).integers(0,CFG['runtime_cases'],(4000,CFG['runtime_cases']))
    med=np.array([np.median(np.load(out/f"runtime_{c['name']}.npz")['milliseconds'],axis=1) for c in CONDITIONS[1:]])
    draw=med[:,:,boot].mean(axis=(0,3))
    ratio_ci={k:np.quantile(draw[0]/draw[j],[.025,.975]).tolist() for k,j in [('vs_full',1),('vs_reference',2)]}
    return {'rows':rows,'decision_mismatches':total_mismatch,'plan_max_abs_error':maxerr,
        'mean_condition_ratio_vs_full':float(np.mean(ratios)),
        'condition_ratios_vs_reference':vsref,'pooled_runtime_ratio_ci95':ratio_ci,
        'runtime_gate':bool(np.mean(ratios)<=1-CFG['runtime_reduction'] and all(r<1 for r in vsref))}

def fixtures():
    rng=np.random.default_rng(89901); q=np.arange(32,dtype=float)[None]/10
    ids=subset(q); assert ids.tolist()==[[0,4,9,13,18,22,27,31]]
    s=summary8(q[:,ids[0]],q[:,ids[0]]+.3)
    assert np.isfinite(s).all() and np.allclose(s[0,:5],[.3,.3,0,.3,.3])
    assert np.allclose(s[0,5:8],0,atol=1e-6) and np.isclose(s[0,8],1)
    sc=np.array([[-.8,-.7,.8,.6,0.,0.,.2]])
    noise=rng.normal(size=(6,1,32,10,2)); m=p.Model(CONDITIONS[-1]); b=loadbank(CONDITIONS[-1])
    mean,sd,inc=p.init_planner(sc)
    for k in range(3):mean,sd,inc,paths,bp,v,best,actions=a2.step(m,sc,mean,sd,inc,noise[k])
    full=old.cost(old.rollout(a2.Corrected(m,b),sc[:,:2],actions),actions,sc)
    ix=subset(v); suba=actions[np.arange(1)[:,None],ix]
    independent=old.cost(old.rollout(a2.Corrected(m,b),sc[:,:2],suba),suba,sc)
    error=float(np.max(np.abs(independent-np.take_along_axis(full,ix,1)))); assert error<1e-12
    j,_,go=controller(np.array([.02,-.02]),np.array([.03,-.01]),np.ones(2),'b8')
    assert np.allclose(j,[.98+.01/12,1+.01/12]) and go.tolist()==[True,False]
    for mode in ['always_stop','always_continue']:
        online_case(m,b,sc,noise,mode,None,None)
    return {'subset_ranks':ids[0].tolist(),'subset_rollout_max_error':error,'finite_subset_summary':True,
        'model_call_accounting':True,'charged_policy':True,'inherited_checks':a2.fixtures()}

def run(resume=False):
    out=ROOT/'run'; hashes=fingerprint()
    if out.exists():
        if not resume:raise RuntimeError('Existing run preserved; use --resume with identical source hashes.')
        manifest=json.loads((out/'manifest.json').read_text()); assert hashes==manifest['hashes']
        if (out/'COMPLETE.json').exists():raise RuntimeError('Run complete; no rerun.')
    else:
        checks=fixtures(); out.mkdir()
        wj(out/'manifest.json',{'study':'F1-A3','version':'main-v1','started_utc_system_clock':datetime.now(timezone.utc).isoformat(),
            'hashes':hashes,'fixtures':checks,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__})
    sc=scenes(); order=np.random.default_rng(CFG['split_seed']).permutation(len(sc)); fit,tune,test=np.split(order,[1200,1600])
    assert len(set(fit)|set(tune)|set(test))==2400
    if not (out/'scenarios.npz').exists():wn(out/'scenarios.npz',scenarios=sc,fit=fit,tune=tune,evaluation=test)
    noise=np.random.default_rng(CFG['noise_seed']).normal(size=(6,len(sc),32,10,2))
    allpred=[]; alltune=[]; ys=[]; bases=[]; refs=[]; policyrefs=[]; selections=[]; grid=[]; diagnostic=[]
    for ci,condition in enumerate(CONDITIONS):
        name=condition['name']; datapath=out/f'data_{name}.npz'; checkpoint=out/f'checkpoint_{name}.json'
        if not datapath.exists():
            data,seconds=generate(p.Model(condition),loadbank(condition),sc,noise)
            wn(datapath,**data); wj(out/f'generation_{name}.json',{'seconds':seconds})
        else:data=dict(np.load(datapath))
        mats,ex,source=matrices(data,fit,[fit,tune,test],ci)
        if not checkpoint.exists():
            wn(out/f'expanders_{name}.npz',**ex); wn(out/f'controls_{name}.npz',source=source)
            selected={}; preds={}; tuning={}; rows=[]; started=time.perf_counter()
            for arm in ARMS:
                best,records=p.fit_arm(mats[arm],data['target'],fit,tune,out/f'observer_{name}_{arm}.npz')
                z=best.pop('predictions'); preds[arm]=z[test]; tuning[arm]=z[tune]
                selected[arm]=dict(best,input_dimensions=mats[arm].shape[1]); rows += [dict(condition=name,arm=arm,**r) for r in records]
            ref=min(REFS,key=lambda a:selected[a]['tuning_mse'])
            rc=min(REFS,key=lambda a:float(controller(tuning[a],data['target'][tune],data['base'][tune],a)[0].mean()))
            wn(out/f'predictions_{name}.npz',predictions=np.array([preds[a] for a in ARMS]),
                tuning_predictions=np.array([tuning[a] for a in ARMS]))
            wj(checkpoint,dict(selected=selected,grid=rows,reference=ref,controller_reference=rc,
                fit_seconds=time.perf_counter()-started))
        saved=json.loads(checkpoint.read_text()); ev=np.load(out/f'predictions_{name}.npz')
        allpred.append(ev['predictions']); alltune.append(ev['tuning_predictions']); ys.append(data['target'][test]); bases.append(data['base'][test])
        refs.append(saved['reference']); policyrefs.append(saved['controller_reference']); selections.append(saved['selected']); grid+=saved['grid']
        diagnostic.append(dict(condition,generation_seconds=json.loads((out/f'generation_{name}.json').read_text())['seconds'],fit_seconds=saved['fit_seconds']))
        assert fingerprint()==hashes
        print(json.dumps({'stage':'condition_checkpoint','condition':name,'reference':refs[-1],'controller_reference':policyrefs[-1]}),flush=True)
    pred=np.array(allpred); y=np.array(ys); base=np.array(bases)
    wn(out/'evaluation.npz',predictions=pred,tuning_predictions=np.array(alltune),target=y,base=base,scene_ids=test)
    wj(out/'candidate_grid.json',grid); assert len(grid)==1008
    result=analyze(pred,y,base,refs,policyrefs)
    result.update(study='F1-A3',version='main-v1',arms=ARMS,conditions=CONDITIONS,references=refs,
        controller_references=policyrefs,selected_observers=selections,offline_diagnostics=diagnostic,
        selected_pipelines=84,candidate_fits=1008)
    wj(out/'result.json',result)
    print(json.dumps({'stage':'headlines','headline':result['headline'],'verdicts':result['primary_verdicts']}),flush=True)
    rt=benchmark(out,sc,noise,test,pred,policyrefs)
    assert rt['decision_mismatches']==0 and rt['plan_max_abs_error']<1e-9 and fingerprint()==hashes
    result.update(runtime=rt,practical_success=bool(all(result['primary_verdicts'].values()) and rt['runtime_gate']))
    wj(out/'result.json',result); wj(out/'COMPLETE.json',{'complete':True,'source_hashes_unchanged':True})
    print(json.dumps({'stage':'finished','practical_success':result['practical_success']}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--fixtures-only',action='store_true'); ap.add_argument('--resume',action='store_true'); args=ap.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures_only:print(json.dumps(fixtures(),indent=2))
        else:run(args.resume)
