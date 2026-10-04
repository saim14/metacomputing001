"""F1-A1 frozen experiment; no neural retraining, no outcome-driven search."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

import numpy as np
import scipy
from scipy.linalg import eigh
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'prior'))
import run_f1 as old
CFG=json.loads((ROOT/'config.json').read_text())
ARMS=['c','ch','cg','chg','cr','chr','cgr','chgr','ce','cre','craw','crraw','cmhg','chmg','cshg','chgmr']
REF0=['c','ce','craw']
REFR=['c','cr','ce','cre','craw','crraw']
CONDITIONS=[{'name':'exact','seed':None,'alpha':0.}]+[
    {'name':f'm{s}_a{int(a*100)}','seed':s,'alpha':a} for s in CFG['model_seeds'] for a in [.5,1.]]


def write_json(path,value):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def write_npz(path,**data):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**data); f.flush(); os.fsync(f.fileno())
    with zipfile.ZipFile(temp) as z:
        assert z.testzip() is None
    os.replace(temp,path)


def fingerprint():
    paths=[ROOT/'study.py',ROOT/'config.json',ROOT/'F1_A1_Protocol.md']
    paths+=sorted((ROOT/'prior').rglob('*'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file() and '__pycache__' not in p.parts}


class Model:
    def __init__(self,condition):
        self.condition=condition
        self.alpha=condition['alpha']
        self.neural=None
        if condition['seed'] is not None:
            with np.load(ROOT/'prior'/'run'/f"world_{condition['seed']}.npz") as w:
                self.neural=old.World(dict(w))

    def __call__(self,x,a):
        if self.alpha==0:return old.physics(x,a)
        if self.alpha==1:return self.neural(x,a)
        true=old.physics(x,a)
        return true+self.alpha*(self.neural(x,a)-true)


class Reliability:
    def __init__(self,x,error2):
        self.scale=np.array([4.,4.,1.,1.])
        self.tree=cKDTree(x/self.scale)
        self.error2=error2
        self.global_rmse=float(np.sqrt(error2.mean()))

    def describe(self,paths,actions):
        n=len(paths)
        queries=np.concatenate([paths[:,:-1],actions],axis=2).reshape(-1,4)
        distances,idx=self.tree.query(queries/self.scale,k=CFG['neighbors'],workers=1)
        local=np.sqrt(self.error2[idx].mean(axis=1)).reshape(n,-1)
        return np.c_[local.mean(1),local.max(1),distances[:,-1].reshape(n,-1).max(1),
                     np.full(n,self.global_rmse)].astype(np.float32)


def scenarios():
    rng=np.random.default_rng(52001); rows=[]
    while len(rows)<CFG['n_scenarios']:
        start,goal=rng.uniform(-1.25,1.25,(2,2))
        if np.linalg.norm(start-goal)<.75:continue
        center=(start+goal)/2+rng.normal(0,.35,2); radius=rng.uniform(.15,.35)
        if min(np.linalg.norm(start-center),np.linalg.norm(goal-center))<=radius+.08:continue
        rows.append(np.r_[start,goal,center,radius])
    return np.array(rows)


def init_planner(sc):
    h=CFG['horizon']
    mean=np.repeat(np.clip((sc[:,2:4]-sc[:,:2])/(.25*h),-.8,.8)[:,None,:],h,axis=1)
    return mean,np.full_like(mean,.65),mean.copy()


def step(model,sc,mean,sd,incumbent,noise):
    n=len(sc)
    actions=np.clip(mean[:,None]+sd[:,None]*noise,-1,1)
    actions[:,0],actions[:,1]=incumbent,mean
    paths=old.rollout(model,sc[:,:2],actions)
    values=old.cost(paths,actions,sc)
    order=np.argsort(values,axis=1,kind='stable')
    chosen=order[:,0]
    incumbent=actions[np.arange(n),chosen].copy()
    best_paths=paths[np.arange(n),chosen].copy()
    best=values[np.arange(n),chosen]
    elite=actions[np.arange(n)[:,None],order[:,:CFG['elites']]]
    mean=elite.mean(1); sd=np.maximum(elite.std(1),.08)
    return mean,sd,incumbent,paths,best_paths,values,best


def current(sc,mean,sd,incumbent,values,best):
    n=len(sc)
    return np.c_[sc,mean.reshape(n,-1),sd.reshape(n,-1),incumbent.reshape(n,-1),np.sort(values,axis=1),best].astype(np.float32)


def features(sc,c,h,paths,best_paths,incumbent,bank,need_g=True,need_r=True,need_raw=True):
    raw_paths=paths.astype(np.float32)
    data={'c':c,'h':h}
    if need_g:data['g']=old.geometry(raw_paths.astype(float),sc.astype(np.float32).astype(float)).astype(np.float32)
    if need_r:data['r']=bank.describe(best_paths.astype(np.float32).astype(float),incumbent.astype(np.float32).astype(float))
    if need_raw:data['raw']=raw_paths.reshape(len(sc),-1)
    return data


def generate(model,bank,sc,noise):
    chunks=[]; started=time.perf_counter()
    for lo in range(0,len(sc),100):
        scene=sc[lo:lo+100]; n=len(scene)
        mean,sd,incumbent=init_planner(scene); history=[]; costs=[]
        for k in range(CFG['rounds']):
            mean,sd,incumbent,paths,best_paths,values,best=step(model,scene,mean,sd,incumbent,noise[k,lo:lo+n])
            costs.append(best.copy())
            if k<CFG['checkpoint']:
                c=current(scene,mean,sd,incumbent,values,best)
                if k<CFG['checkpoint']-1:history.append(c.copy())
                else:
                    at_checkpoint=features(scene,c,np.stack(history,axis=1),paths,best_paths,incumbent,bank)
                    u3=incumbent.copy()
        estimated=np.stack(costs,axis=1)
        assert np.max(np.diff(estimated,axis=1))<1e-9
        base=old.cost(old.rollout(old.physics,scene[:,:2],u3[:,None]),u3[:,None],scene)[:,0]
        final=old.cost(old.rollout(old.physics,scene[:,:2],incumbent[:,None]),incumbent[:,None],scene)[:,0]
        at_checkpoint.update(u3=u3,u6=incumbent.copy(),base=base,final=final,target=base-final,estimated_costs=estimated)
        chunks.append(at_checkpoint)
    return {key:np.concatenate([chunk[key] for chunk in chunks]) for key in chunks[0]},time.perf_counter()-started


def standardize(x,idx):
    mean=x[idx].mean(0); sd=x[idx].std(0); sd[sd<1e-9]=1
    return (x-mean)/sd,mean,sd


def expansion(x,mean,sd):
    rng=np.random.default_rng(54001)
    w=rng.normal(size=(x.shape[1],CFG['expanded_width']))/np.sqrt(x.shape[1])
    phase=rng.uniform(0,2*np.pi,CFG['expanded_width'])
    return (np.sqrt(2/CFG['expanded_width'])*np.cos(((x-mean)/sd)@w+phase)).astype(np.float32)


def deranged_rows(values,splits,rng):
    out=np.empty_like(values); mappings=np.empty(len(values),dtype=int)
    for ids in splits:
        order=rng.permutation(ids); src=np.roll(order,1)
        assert np.all(src!=order)
        out[order]=values[src]; mappings[order]=src
    return out,mappings


def arm_matrices(data,fit,splits,condition_index):
    c,h,g,r,raw=[data[k].astype(float) for k in ['c','h','g','r','raw']]
    h=h.reshape(len(c),-1); cr=np.c_[c,r]
    _,cm,cs=standardize(c,fit); _,rm,rs=standardize(cr,fit)
    rng=np.random.default_rng(54002+condition_index)
    mh,ih=deranged_rows(h,splits,rng); mg,ig=deranged_rows(g,splits,rng); mr,ir=deranged_rows(r,splits,rng)
    sh=h.reshape(len(c),2,100).copy()
    swaps=np.random.default_rng(54003+condition_index).integers(0,2,len(c)).astype(bool)
    sh[swaps]=sh[swaps,::-1]; sh=sh.reshape(len(c),-1)
    matrices={'c':c,'ch':np.c_[c,h],'cg':np.c_[c,g],'chg':np.c_[c,h,g],
              'cr':cr,'chr':np.c_[c,h,r],'cgr':np.c_[c,g,r],'chgr':np.c_[c,h,g,r],
              'ce':np.c_[c,expansion(c,cm,cs)],'cre':np.c_[cr,expansion(cr,rm,rs)],
              'craw':np.c_[c,raw],'crraw':np.c_[c,r,raw],
              'cmhg':np.c_[c,mh,g],'chmg':np.c_[c,h,mg],'cshg':np.c_[c,sh,g],'chgmr':np.c_[c,h,g,mr]}
    matrices={key:value.astype(np.float32).astype(float) for key,value in matrices.items()}
    return matrices,{'cm':cm,'cs':cs,'rm':rm,'rs':rs},{'h_source':ih,'g_source':ig,'r_source':ir,'past_swapped':swaps}


def basis(z,index):
    if index==0:return z
    bandwidth=CFG['bandwidths'][index-1]
    rng=np.random.default_rng(55001)
    w=rng.normal(size=(z.shape[1],CFG['rff_width']))/(np.sqrt(z.shape[1])*bandwidth)
    phase=rng.uniform(0,2*np.pi,CFG['rff_width'])
    return np.sqrt(2/CFG['rff_width'])*np.cos(z@w+phase)


def fit_arm(x,y,fit,tune,path):
    z,xm,xs=standardize(x,fit); ym=y[fit].mean()
    state={'xm':xm,'xs':xs,'ym':np.array(ym)}; records=[]; best=None
    for bi in range(3):
        b,bm,bs=standardize(basis(z,bi),fit)
        vals,vecs=eigh(b[fit].T@b[fit],check_finite=False); vals=np.maximum(vals,0)
        rhs=vecs.T@(b[fit].T@(y[fit]-ym)); betas=[]
        for ai,alpha in enumerate(CFG['alphas']):
            beta=vecs@(rhs/(vals+len(fit)*alpha)); betas.append(beta)
            pred=b@beta+ym; mse=float(np.mean((pred[tune]-y[tune])**2))
            row={'basis_index':bi,'alpha_index':ai,'alpha':alpha,'tuning_mse':mse,'coefficients':len(beta)+1}
            records.append(row)
            if best is None or mse<best['tuning_mse']:best=dict(row,predictions=pred.copy())
        state[f'b{bi}_mean']=bm; state[f'b{bi}_sd']=bs; state[f'b{bi}_beta']=np.stack(betas,axis=1)
    state['selected_basis']=np.array(best['basis_index']); state['selected_alpha']=np.array(best['alpha_index'])
    write_npz(path,**state)
    return best,records


class Predictor:
    def __init__(self,path):
        with np.load(path) as p:self.state=dict(p)
        s=self.state; self.bi=int(s['selected_basis']); self.ai=int(s['selected_alpha'])
        self.w=self.phase=None
        if self.bi:
            d=len(s['xm']); rng=np.random.default_rng(55001)
            self.w=rng.normal(size=(d,CFG['rff_width']))/(np.sqrt(d)*CFG['bandwidths'][self.bi-1])
            self.phase=rng.uniform(0,2*np.pi,CFG['rff_width'])

    def __call__(self,x):
        s=self.state; z=(x-s['xm'])/s['xs']
        if self.bi:z=np.sqrt(2/CFG['rff_width'])*np.cos(z@self.w+self.phase)
        z=(z-s[f'b{self.bi}_mean'])/s[f'b{self.bi}_sd']
        return z@s[f'b{self.bi}_beta'][:,self.ai]+s['ym']


def intervals(values,boot):
    values=np.asarray(values)
    v=values.mean(axis=0) if values.ndim==2 else values
    samples=v[boot].mean(1)
    return {'mean':float(v.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'ci9875':np.quantile(samples,[.00625,.99375]).tolist()}


def mse_comparison(reference,candidate,y,boot):
    er=(reference-y)**2; ec=(candidate-y)**2
    out=intervals(er-ec,boot)
    out.update(reference_mse=float(er.mean()),candidate_mse=float(ec.mean()),relative_reduction=float(1-ec.mean()/max(er.mean(),1e-15)))
    return out


def controller(pred,y,base,charge):
    keep=pred>charge
    loss=base-y*keep
    return loss+charge*keep,loss,keep


def analyze(prediction,y,base,fitmeans,refs0,refsr,controller_refs,indices):
    boot=np.random.default_rng(56001).integers(0,y.shape[1],(CFG['bootstrap'],y.shape[1]))
    p={arm:prediction[:,i] for i,arm in enumerate(ARMS)}
    r0=np.array([p[name][i] for i,name in enumerate(refs0)])
    rr=np.array([p[name][i] for i,name in enumerate(refsr)])
    pc=np.array([p[name][i] for i,name in enumerate(controller_refs)])
    imperfect=slice(1,None)
    full_ids=[i for i,c in enumerate(CONDITIONS) if c['alpha']==1]
    harm=(y<-.01).astype(float)
    intervention=intervals(harm[full_ids]-harm[0],boot)
    joint=mse_comparison(r0[imperfect],p['chg'][imperfect],y[imperfect],boot)
    reliability=mse_comparison(p['chg'][imperfect],p['chgr'][imperfect],y[imperfect],boot)
    charge=CFG['charge']
    cand_j,cand_l,cand_go=controller(p['chgr'],y,base,charge)
    ref_j,ref_l,ref_go=controller(pc,y,base,charge)
    decision=intervals((ref_j-cand_j)[imperfect],boot)
    decision.update(reference_objective=float(ref_j[imperfect].mean()),candidate_objective=float(cand_j[imperfect].mean()),
                    relative_reduction=float(1-cand_j[imperfect].mean()/ref_j[imperfect].mean()))
    diagnostics={a:mse_comparison(p[a][imperfect],p['chg'][imperfect],y[imperfect],boot) for a in ['ch','cg','cmhg','chmg','cshg']}
    diagnostics['reliability_mismatch']=mse_comparison(p['chgmr'][imperfect],p['chgr'][imperfect],y[imperfect],boot)
    diagnostics['combined_vs_current_reliability']=mse_comparison(rr[imperfect],p['chgr'][imperfect],y[imperfect],boot)
    verdicts={
      'error_intervention':intervention['mean']>=.05 and intervention['ci9875'][0]>0,
      'joint_history_geometry':joint['relative_reduction']>=.05 and joint['ci9875'][0]>0,
      'added_reliability':reliability['relative_reduction']>=.05 and reliability['ci9875'][0]>0 and diagnostics['reliability_mismatch']['ci95'][0]>0,
      'charged_controller':decision['relative_reduction']>=.01 and decision['ci9875'][0]>0}
    per=[]
    for i,c in enumerate(CONDITIONS):
        oracle_j=base[i]-np.maximum(y[i]-charge,0)
        per.append(dict(c,harmful_fraction=float(harm[i].mean()),value_mean=float(y[i].mean()),
                         arm_mse={a:float(np.mean((p[a][i]-y[i])**2)) for a in ARMS},
                         constant_mse=float(np.mean((fitmeans[i]-y[i])**2)),
                         reference_no_r=refs0[i],reference_with_r=refsr[i],controller_reference=controller_refs[i],
                         candidate_charged_cost=float(cand_j[i].mean()),reference_charged_cost=float(ref_j[i].mean()),
                         candidate_true_cost=float(cand_l[i].mean()),reference_true_cost=float(ref_l[i].mean()),
                         candidate_continue_fraction=float(cand_go[i].mean()),reference_continue_fraction=float(ref_go[i].mean()),
                         candidate_harmful_continuation_fraction=float((cand_go[i]&(y[i]<-.01)).mean()),
                         reference_harmful_continuation_fraction=float((ref_go[i]&(y[i]<-.01)).mean()),
                         stop_cost=float(base[i].mean()),continue_cost=float((base[i]-y[i]+charge).mean()),oracle_cost=float(oracle_j.mean())))
    sensitivity={}
    for c in [0.,.005,.01,.02,.05]:
        # References stay fixed from primary-charge tuning; no test-dependent reselection.
        cj,_,cg=controller(p['chgr'],y,base,c); rj,_,rg=controller(pc,y,base,c)
        sensitivity[str(c)]={'difference':intervals((rj-cj)[imperfect],boot),
                             'candidate_continue_fraction':float(cg[imperfect].mean()),'reference_continue_fraction':float(rg[imperfect].mean())}
    return {'headline':{'error_intervention':intervention,'joint_history_geometry':joint,'added_reliability':reliability,'charged_controller':decision},
            'verdicts':verdicts,'descriptive_comparisons':diagnostics,'per_condition':per,'charge_sensitivity':sensitivity,
            'pooled_imperfect_arm_mse':{a:float(np.mean((p[a][imperfect]-y[imperfect])**2)) for a in ARMS},
            'component_attribution_supported':all(diagnostics[a]['ci95'][0]>0 for a in ['ch','cg','cmhg','chmg'])}


def online_features(arm,sc,mean,sd,inc,paths,best_paths,values,best,history,bank,expander):
    c=current(sc,mean,sd,inc,values,best).astype(float)
    r=bank.describe(best_paths.astype(np.float32).astype(float),inc.astype(np.float32).astype(float)).astype(float) if arm in ['cr','cre','crraw','chgr'] else None
    if arm=='c':return c
    if arm=='cr':return np.c_[c,r]
    if arm=='ce':return np.c_[c,expansion(c,expander['cm'],expander['cs'])].astype(np.float32).astype(float)
    if arm=='cre':
        cr=np.c_[c,r]
        return np.c_[cr,expansion(cr,expander['rm'],expander['rs'])].astype(np.float32).astype(float)
    if arm=='craw':return np.c_[c,paths.astype(np.float32).reshape(len(sc),-1)]
    if arm=='crraw':return np.c_[c,r,paths.astype(np.float32).reshape(len(sc),-1)]
    if arm=='chgr':
        h=np.stack(history,axis=1).reshape(len(sc),-1).astype(float)
        g=old.geometry(paths.astype(np.float32).astype(float),sc.astype(np.float32).astype(float)).astype(np.float32)
        return np.c_[c,h,g,r]
    raise ValueError(arm)


def online_case(model,bank,sc,noise,mode,predictor,expander):
    mean,sd,inc=init_planner(sc); history=[]; continued=mode=='always_continue'
    for k in range(CFG['rounds']):
        mean,sd,inc,paths,bp,values,best=step(model,sc,mean,sd,inc,noise[k:k+1][0])
        if mode=='chgr' and k<2:history.append(current(sc,mean,sd,inc,values,best))
        if k+1==CFG['checkpoint']:
            if mode=='always_stop':break
            if mode!='always_continue':
                x=online_features(mode,sc,mean,sd,inc,paths,bp,values,best,history,bank,expander)
                continued=bool(predictor(x)[0]>CFG['charge'])
                if not continued:break
    return inc,continued


def benchmark(output,sc,noise,test,prediction,controller_refs):
    cal=dict(np.load(output/'calibration_shared.npz')); all_rows=[]; mismatch=0; max_plan_error=0.
    for ci,condition in enumerate(CONDITIONS):
        name=condition['name']; model=Model(condition)
        bank=Reliability(cal['x'],np.load(output/f'calibration_{name}.npz')['error2'])
        expander=dict(np.load(output/f'expanders_{name}.npz'))
        cache=dict(np.load(output/f'data_{name}.npz'))
        reference=controller_refs[ci]
        modes=['chgr',reference,'always_stop','always_continue']
        predictors={m:Predictor(output/f'observer_{name}_{m}.npz') for m in ['chgr',reference]}
        ids=test[:CFG['runtime_cases']]
        # Warm up every method using the first fixed evaluation scene.
        for mode in modes:online_case(model,bank,sc[ids[:1]],noise[:,ids[:1]],mode,predictors.get(mode),expander)
        totals={m:[] for m in modes}
        for rep in range(CFG['runtime_repeats']):
            rotated=modes[rep:]+modes[:rep]
            for mode in rotated:
                elapsed=0.
                for pos,sid in enumerate(ids):
                    t=time.perf_counter()
                    plan,go=online_case(model,bank,sc[sid:sid+1],noise[:,sid:sid+1],mode,predictors.get(mode),expander)
                    elapsed+=time.perf_counter()-t
                    expected=(mode=='always_continue') if mode.startswith('always_') else bool(prediction[ci,ARMS.index(mode),pos]>CFG['charge'])
                    mismatch+=int(go!=expected)
                    expected_plan=cache['u6' if expected else 'u3'][sid:sid+1]
                    max_plan_error=max(max_plan_error,float(np.max(np.abs(plan-expected_plan))))
                totals[mode].append(elapsed*1000/len(ids))
        row={'condition':name,'cases':len(ids),'repeats':CFG['runtime_repeats'],'ms_per_case':{m:float(np.median(v)) for m,v in totals.items()},'repetitions_ms':totals}
        all_rows.append(row); print(json.dumps({'stage':'runtime',**row}),flush=True)
    return {'rows':all_rows,'decision_mismatches':mismatch,'plan_max_abs_error':max_plan_error,
            'runtime_savings_supported':all(r['ms_per_case']['chgr']<r['ms_per_case']['always_continue'] for r in all_rows[1:])}


def fixtures():
    rng=np.random.default_rng(999101); x=rng.uniform(-1,1,(10,2)); a=rng.uniform(-1,1,(10,2))
    exact=Model(CONDITIONS[0]); half=Model(CONDITIONS[1]); full=Model(CONDITIONS[2])
    error=float(np.max(np.abs((half(x,a)-exact(x,a))-.5*(full(x,a)-exact(x,a)))))
    assert error<1e-12 and np.array_equal(exact(x,a),old.physics(x,a))
    arr=np.arange(20)[:,None]; parts=[np.arange(12),np.arange(12,20)]
    shuffled,idx=deranged_rows(arr,parts,rng)
    assert np.all(idx!=np.arange(20)) and all(set(idx[p])==set(p) for p in parts)
    j,loss,go=controller(np.array([.02,-.02,.005]),np.array([.03,-.01,.004]),np.ones(3),.01)
    assert np.array_equal(go,[True,False,False]) and np.allclose(j,[.98,1,1])
    return {'interpolation_max_error':error,'strict_derangement':True,'fixed_charge_controller':True}


def run():
    output=ROOT/'run'
    if output.exists():raise RuntimeError('Preserve the existing run; do not silently rerun.')
    output.mkdir()
    frozen=fingerprint()
    write_json(output/'manifest.json',{'study':'F1-A1','version':'1.0','started_utc':datetime.now(timezone.utc).isoformat(),
        'hashes':frozen,'fixtures':fixtures(),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__})
    sc=scenarios(); order=np.random.default_rng(52002).permutation(len(sc))
    fit,tune,test=np.split(order,[CFG['n_fit'],CFG['n_fit']+CFG['n_tune']])
    assert len(set(fit)|set(tune)|set(test))==len(sc) and not(set(fit)&set(test))
    write_npz(output/'scenarios.npz',scenarios=sc,fit=fit,tune=tune,evaluation=test)
    noise=np.random.default_rng(53001).normal(size=(CFG['rounds'],len(sc),CFG['candidates'],CFG['horizon'],2))
    rng=np.random.default_rng(51001)
    calx=np.c_[rng.uniform(-4,4,(CFG['calibration_size'],2)),rng.uniform(-1,1,(CFG['calibration_size'],2))]
    caltrue=old.physics(calx[:,:2],calx[:,2:])
    write_npz(output/'calibration_shared.npz',x=calx,true_next=caltrue)
    diagnostic_rng=np.random.default_rng(51002)
    starts=diagnostic_rng.uniform(-1.25,1.25,(256,2)); actions=diagnostic_rng.uniform(-1,1,(256,1,10,2))
    true_paths=old.rollout(old.physics,starts,actions)
    all_predictions=[]; all_tuning=[]; all_y=[]; all_base=[]; fitmeans=[]; references0=[]; referencesr=[]; controller_refs=[]
    selections=[]; candidate_rows=[]; world_stats=[]
    for ci,condition in enumerate(CONDITIONS):
        name=condition['name']; model=Model(condition)
        t=time.perf_counter(); error2=np.mean((model(calx[:,:2],calx[:,2:])-caltrue)**2,axis=1)
        bank=Reliability(calx,error2); calibration_seconds=time.perf_counter()-t
        write_npz(output/f'calibration_{name}.npz',error2=error2)
        rmse=float(np.sqrt(np.mean((old.rollout(model,starts,actions)[:,:,1:]-true_paths[:,:,1:])**2)))
        data,seconds=generate(model,bank,sc,noise)
        write_npz(output/f'data_{name}.npz',**data)
        world_stats.append(dict(condition,one_step_rmse=bank.global_rmse,ten_step_rmse=rmse,calibration_seconds=calibration_seconds,generation_seconds=seconds))
        print(json.dumps({'stage':'planning',**world_stats[-1],'value_sd_fit':float(data['target'][fit].std()),'value_sd_tune':float(data['target'][tune].std())}),flush=True)
        matrices,expander,controls=arm_matrices(data,fit,[fit,tune,test],ci)
        write_npz(output/f'expanders_{name}.npz',**expander)
        write_npz(output/f'controls_{name}.npz',**controls)
        chosen={}; preds={}; tune_preds={}
        for arm in ARMS:
            best,grid=fit_arm(matrices[arm],data['target'],fit,tune,output/f'observer_{name}_{arm}.npz')
            pred=best.pop('predictions'); preds[arm]=pred[test]; tune_preds[arm]=pred[tune]
            chosen[arm]=dict(best,input_dimensions=matrices[arm].shape[1])
            candidate_rows.extend([dict(condition=name,arm=arm,**row) for row in grid])
        r0=min(REF0,key=lambda arm:chosen[arm]['tuning_mse']); rr=min(REFR,key=lambda arm:chosen[arm]['tuning_mse'])
        rc=min(REFR,key=lambda arm:float(controller(tune_preds[arm],data['target'][tune],data['base'][tune],CFG['charge'])[0].mean()))
        references0.append(r0); referencesr.append(rr); controller_refs.append(rc); selections.append(chosen)
        all_predictions.append(np.array([preds[a] for a in ARMS])); all_tuning.append(np.array([tune_preds[a] for a in ARMS]))
        all_y.append(data['target'][test]); all_base.append(data['base'][test]); fitmeans.append(float(data['target'][fit].mean()))
        print(json.dumps({'stage':'observers','condition':name,'selected_no_r':r0,'selected_with_r':rr,'controller_reference':rc}),flush=True)
    prediction=np.array(all_predictions); y=np.array(all_y); base=np.array(all_base)
    write_npz(output/'evaluation.npz',predictions=prediction,tuning_predictions=np.array(all_tuning),target=y,base=base,scenario_ids=test,fit_means=np.array(fitmeans))
    write_json(output/'candidate_grid.json',candidate_rows)
    result=analyze(prediction,y,base,np.array(fitmeans),references0,referencesr,controller_refs,test)
    result.update(study='F1-A1',version='1.0',conditions=CONDITIONS,arms=ARMS,world_diagnostics=world_stats,
                  selected_observers=selections,candidate_fits=len(candidate_rows),selected_pipelines=len(CONDITIONS)*len(ARMS),
                  primary_refs=references0,reliability_refs=referencesr,controller_refs=controller_refs)
    write_json(output/'result.json',result)
    assert len(candidate_rows)==1344
    print(json.dumps({'stage':'headlines','verdicts':result['verdicts'],'headline':result['headline']}),flush=True)
    runtime=benchmark(output,sc,noise,test,prediction,controller_refs)
    write_json(output/'runtime.json',runtime)
    assert runtime['decision_mismatches']==0 and runtime['plan_max_abs_error']<1e-9
    assert fingerprint()==frozen
    result['runtime']=runtime; result['finished_utc']=datetime.now(timezone.utc).isoformat()
    write_json(output/'result.json',result)
    print(json.dumps({'stage':'finished','verdicts':result['verdicts'],'runtime_verified':True}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--fixtures-only',action='store_true'); args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures_only:print(json.dumps(fixtures()))
        else:run()
