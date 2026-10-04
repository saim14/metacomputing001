"""F1-A2 frozen development experiment. No outcome-driven extensions."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.spatial import cKDTree
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('f1_previous',ROOT/'prior/study.py')
prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)
old=prev.old
CFG=json.loads((ROOT/'config.json').read_text())
CONDITIONS=prev.CONDITIONS
ARMS=['c','cr','ce','cre','craw','crraw','x','cp','cb','xp','xb','cmp','cmb','cpi','cbi']
REFS=['c','cr','ce','cre','craw','crraw','x']
B_NAMES=['incumbent_bias','mean_bias','sd_bias','min_bias','max_bias',
         'elite_relative_bias','rest_relative_bias','bias_cost_slope',
         'cost_correlation','discordant_pair_fraction','candidate_set_regret','better_candidate_fraction']
write_json=prev.write_json
write_npz=prev.write_npz
standardize=prev.standardize


def fingerprint():
    paths=[ROOT/'study.py',ROOT/'config.json',ROOT/'F1_A2_Protocol.md']+sorted((ROOT/'prior').rglob('*'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file() and '__pycache__' not in p.parts}


class Bank:
    def __init__(self,x,residual):
        self.scale=np.array([4.,4.,1.,1.]);self.x=x;self.residual=residual
        self.tree=cKDTree(x/self.scale)
        self.magnitude=prev.Reliability(x,np.mean(residual**2,axis=1))
    def correct(self,x,a):
        shape=x.shape;q=np.concatenate([x,a],axis=-1).reshape(-1,4)/self.scale
        dist,idx=self.tree.query(q,k=CFG['neighbors'],workers=1)
        radius=np.maximum(dist[:,-1],1e-9)
        offset=(self.x[idx]/self.scale-q[:,None])/radius[:,None,None]
        design=np.concatenate([np.ones((*offset.shape[:2],1)),offset],axis=2)
        weights=np.exp(-.5*(dist/radius[:,None])**2)
        gram=np.einsum('nki,nkj,nk->nij',design,design,weights)
        gram+=np.diag([0.]+[CFG['local_ridge']]*4)[None]
        rhs=np.einsum('nki,nkj,nk->nij',design,self.residual[idx],weights)
        coef=np.linalg.solve(gram,rhs)
        return coef[:,0].reshape(shape)
    def describe(self,paths,actions):
        return self.magnitude.describe(paths,actions)


class Corrected:
    def __init__(self,model,bank):self.model=model;self.bank=bank
    def __call__(self,x,a):return self.model(x,a)+self.bank.correct(x,a)


def scenarios():
    rng=np.random.default_rng(CFG['scene_seed']);rows=[]
    while len(rows)<CFG['n_scenarios']:
        start,goal=rng.uniform(-1.25,1.25,(2,2))
        if np.linalg.norm(start-goal)<.75:continue
        center=(start+goal)/2+rng.normal(0,.35,2);radius=rng.uniform(.15,.35)
        if min(np.linalg.norm(start-center),np.linalg.norm(goal-center))<=radius+.08:continue
        rows.append(np.r_[start,goal,center,radius])
    return np.array(rows)


def step(model,sc,mean,sd,inc,noise):
    actions=np.clip(mean[:,None]+sd[:,None]*noise,-1,1)
    actions[:,0],actions[:,1]=inc,mean
    result=prev.step(model,sc,mean,sd,inc,noise)
    return (*result,actions)


def bias_summary(q,t):
    n,k=q.shape;b=t-q;row=np.arange(n);order=np.argsort(q,axis=1,kind='stable');chosen=order[:,0]
    incumbent=b[row,chosen];tc=t[row,chosen]
    qc=q-q.mean(1,keepdims=True);bc=b-b.mean(1,keepdims=True);cc=t-t.mean(1,keepdims=True)
    pairs=np.triu_indices(k,1)
    discord=((q[:,pairs[0]]-q[:,pairs[1]])*(t[:,pairs[0]]-t[:,pairs[1]])<0).mean(1)
    elite=np.take_along_axis(b,order[:,:CFG['elites']],axis=1).mean(1)-incumbent
    rest=np.take_along_axis(b,order[:,CFG['elites']:],axis=1).mean(1)-incumbent
    return np.stack([incumbent,b.mean(1),b.std(1),b.min(1),b.max(1),elite,rest,
          (qc*bc).mean(1)/(np.mean(qc**2,axis=1)+1e-12),
          (qc*cc).mean(1)/np.maximum(np.sqrt(np.mean(qc**2,axis=1)*np.mean(cc**2,axis=1)),1e-12),
          discord,tc-t.min(1),(t<tc[:,None]-.01).mean(1)],axis=1).astype(np.float32)


def generate(model,bank,sc,noise):
    chunks=[];started=time.perf_counter();corrected=Corrected(model,bank)
    for lo in range(0,len(sc),100):
        scene=sc[lo:lo+100];n=len(scene);mean,sd,inc=prev.init_planner(scene);history=[];estimated=[]
        for k in range(CFG['rounds']):
            mean,sd,inc,paths,bp,values,best,actions=step(model,scene,mean,sd,inc,noise[k,lo:lo+n])
            estimated.append(best.copy())
            if k<CFG['checkpoint']:
                c=prev.current(scene,mean,sd,inc,values,best)
                if k<CFG['checkpoint']-1:history.append(c.copy())
                else:
                    prefix=prev.features(scene,c,np.stack(history,axis=1),paths,bp,inc,bank)
                    true_cost=old.cost(old.rollout(old.physics,scene[:,:2],actions),actions,scene)
                    corrected_cost=old.cost(old.rollout(corrected,scene[:,:2],actions),actions,scene)
                    prefix.update(p=bias_summary(values,true_cost),b=bias_summary(values,corrected_cost),
                                  model_cost=values.copy(),true_candidate_cost=true_cost,
                                  corrected_candidate_cost=corrected_cost,actions3=actions.copy())
                    u3=inc.copy()
        estimated=np.stack(estimated,axis=1);assert np.max(np.diff(estimated,axis=1))<1e-9
        base=old.cost(old.rollout(old.physics,scene[:,:2],u3[:,None]),u3[:,None],scene)[:,0]
        final=old.cost(old.rollout(old.physics,scene[:,:2],inc[:,None]),inc[:,None],scene)[:,0]
        prefix.update(u3=u3,u6=inc.copy(),base=base,final=final,target=base-final,estimated_costs=estimated)
        chunks.append(prefix)
    return {key:np.concatenate([ch[key] for ch in chunks]) for key in chunks[0]},time.perf_counter()-started


def expansion(x,mean,sd):
    rng=np.random.default_rng(CFG['expansion_seed']);d=CFG['expansion_width']
    w=rng.normal(size=(x.shape[1],d))/np.sqrt(x.shape[1]);phase=rng.uniform(0,2*np.pi,d)
    return (np.sqrt(2/d)*np.cos(((x-mean)/sd)@w+phase)).astype(np.float32)


def arm_matrices(data,fit,splits,ci):
    c,h,g,r,raw,p,b=[data[k].astype(float) for k in ['c','h','g','r','raw','p','b']]
    cr=np.c_[c,r];x=np.c_[c,h.reshape(len(c),-1),g,r]
    _,cm,cs=standardize(c,fit);_,rm,rs=standardize(cr,fit)
    rng=np.random.default_rng(CFG['mismatch_seed']+ci)
    mp,ip=prev.deranged_rows(p,splits,rng);mb,ib=prev.deranged_rows(b,splits,rng)
    mats={'c':c,'cr':cr,'ce':np.c_[c,expansion(c,cm,cs)],'cre':np.c_[cr,expansion(cr,rm,rs)],
          'craw':np.c_[c,raw],'crraw':np.c_[cr,raw],'x':x,'cp':np.c_[c,p],'cb':np.c_[c,b],
          'xp':np.c_[x,p],'xb':np.c_[x,b],'cmp':np.c_[c,mp],'cmb':np.c_[c,mb],
          'cpi':np.c_[c,p[:,0]],'cbi':np.c_[c,b[:,0]]}
    return ({a:v.astype(np.float32).astype(float) for a,v in mats.items()},
            {'cm':cm,'cs':cs,'rm':rm,'rs':rs},{'p_source':ip,'b_source':ib})


def intervals(v,boot):
    v=v.mean(0) if v.ndim==2 else v;samples=v[boot].mean(1);tail=.05/6
    return {'mean':float(v.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'ci_adjusted':np.quantile(samples,[tail,1-tail]).tolist()}


def comparison(reference,candidate,y,boot):
    er=(reference-y)**2;ec=(candidate-y)**2;out=intervals(er-ec,boot)
    out.update(reference_mse=float(er.mean()),candidate_mse=float(ec.mean()),
               relative_reduction=float(1-ec.mean()/max(er.mean(),1e-15)))
    return out


def call_fraction(arm):
    return 1/3 if arm in ['cb','xb','cmb'] else (10/960 if arm=='cbi' else 0.)


def controller(pred,y,base,charge,arm):
    go=pred>charge;true=base-y*go
    return true+charge*(go.astype(float)+call_fraction(arm)),true,go


def analyze(pred,y,base,refs,controller_refs):
    boot=np.random.default_rng(CFG['bootstrap_seed']).integers(0,y.shape[1],(CFG['bootstrap'],y.shape[1]))
    p={a:pred[:,i] for i,a in enumerate(ARMS)};s=slice(1,None)
    ref=np.array([p[a][i] for i,a in enumerate(refs)])
    policy_ref=np.array([p[a][i] for i,a in enumerate(controller_refs)])
    privileged=comparison(ref[s],p['cp'][s],y[s],boot);estimated=comparison(ref[s],p['cb'][s],y[s],boot)
    charge=CFG['charge'];cj,cl,cg=controller(p['cb'],y,base,charge,'cb')
    rj,rl,rg=controller(policy_ref,y,base,charge,'c')
    policy=intervals((rj-cj)[s],boot)
    policy.update(reference_objective=float(rj[s].mean()),candidate_objective=float(cj[s].mean()),
                  relative_reduction=float(1-cj[s].mean()/rj[s].mean()))
    pairs={'paired_p':('cmp','cp'),'paired_b':('cmb','cb'),'set_beyond_incumbent_p':('cpi','cp'),
           'set_beyond_incumbent_b':('cbi','cb'),'estimated_vs_magnitude':('cr','cb'),
           'integration_privileged':('cp','xp'),'integration_estimated':('cb','xb'),
           'added_bias_to_x':('x','xb'),'privileged_estimated_gap':('cb','cp')}
    descriptive={key:comparison(p[a][s],p[b][s],y[s],boot) for key,(a,b) in pairs.items()}
    passed={'privileged_signal':privileged['relative_reduction']>=.05 and privileged['ci_adjusted'][0]>0,
            'estimated_signal':estimated['relative_reduction']>=.05 and estimated['ci_adjusted'][0]>0,
            'charged_controller':policy['relative_reduction']>=.01 and policy['ci_adjusted'][0]>0}
    rows=[]
    for i,condition in enumerate(CONDITIONS):
        row=dict(condition,value_mean=float(y[i].mean()),harmful_fraction=float((y[i]<-.01).mean()),
                 reference=refs[i],controller_reference=controller_refs[i],
                 mse={a:float(np.mean((p[a][i]-y[i])**2)) for a in ARMS},
                 always_stop=float(base[i].mean()),always_continue=float((base[i]-y[i]+charge).mean()),
                 hindsight_oracle=float((base[i]-np.maximum(y[i]-charge,0)).mean()),controllers={})
        for arm in ['cp','cb','xp','xb',controller_refs[i]]:
            j,l,g=controller(p[arm][i],y[i],base[i],charge,arm)
            row['controllers'][arm]={'charged_cost':float(j.mean()),'true_cost':float(l.mean()),
                'continue_fraction':float(g.mean()),'harmful_continuation_fraction':float((g&(y[i]<-.01)).mean()),
                'feature_call_charge':charge*call_fraction(arm)}
        rows.append(row)
    sensitivity={}
    for charge in [0.,.005,.01,.02,.05]:
        cj,_,cg=controller(p['cb'],y,base,charge,'cb');rj,_,rg=controller(policy_ref,y,base,charge,'c')
        sensitivity[str(charge)]={'difference':intervals((rj-cj)[s],boot),
             'candidate_continue_fraction':float(cg[s].mean()),'reference_continue_fraction':float(rg[s].mean())}
    return {'headline':{'privileged_signal':privileged,'estimated_signal':estimated,'charged_controller':policy},
            'verdicts':passed,'descriptive_comparisons':descriptive,'per_condition':rows,
            'pooled_arm_mse':{a:float(np.mean((p[a][s]-y[s])**2)) for a in ARMS},
            'charge_sensitivity':sensitivity,'adjusted_confidence_level':1-.05/3,
            'paired_p_attribution':descriptive['paired_p']['ci95'][0]>0,
            'paired_b_attribution':descriptive['paired_b']['ci95'][0]>0,
            'candidate_set_p_attribution':descriptive['set_beyond_incumbent_p']['ci95'][0]>0,
            'candidate_set_b_attribution':descriptive['set_beyond_incumbent_b']['ci95'][0]>0}


def online_input(arm,sc,mean,sd,inc,paths,bp,values,best,actions,history,bank,model,expander):
    c=prev.current(sc,mean,sd,inc,values,best).astype(float)
    r=bank.describe(bp.astype(np.float32).astype(float),inc.astype(np.float32).astype(float)).astype(float) if arm in ['cr','cre','crraw','x','xb'] else None
    if arm=='c':return c
    if arm=='cr':return np.c_[c,r]
    if arm=='ce':return np.c_[c,expansion(c,expander['cm'],expander['cs'])].astype(np.float32).astype(float)
    if arm=='cre':
        cr=np.c_[c,r];return np.c_[cr,expansion(cr,expander['rm'],expander['rs'])].astype(np.float32).astype(float)
    if arm=='craw':return np.c_[c,paths.astype(np.float32).reshape(len(sc),-1)]
    if arm=='crraw':return np.c_[c,r,paths.astype(np.float32).reshape(len(sc),-1)]
    if arm in ['x','xb']:
        h=np.stack(history,axis=1).reshape(len(sc),-1).astype(float)
        g=old.geometry(paths.astype(np.float32).astype(float),sc.astype(np.float32).astype(float)).astype(np.float32)
        x=np.c_[c,h,g,r]
        if arm=='x':return x
    if arm in ['cb','xb']:
        costs=old.cost(old.rollout(Corrected(model,bank),sc[:,:2],actions),actions,sc)
        b=bias_summary(values,costs).astype(float)
        return np.c_[c if arm=='cb' else x,b]
    raise ValueError(arm)


def online_case(model,bank,scene,noise,mode,predictor,expander):
    mean,sd,inc=prev.init_planner(scene);history=[];go=mode=='always_continue'
    for k in range(CFG['rounds']):
        mean,sd,inc,paths,bp,values,best,actions=step(model,scene,mean,sd,inc,noise[k])
        if mode in ['x','xb'] and k<2:history.append(prev.current(scene,mean,sd,inc,values,best))
        if k+1==CFG['checkpoint']:
            if mode=='always_stop':break
            if mode!='always_continue':
                x=online_input(mode,scene,mean,sd,inc,paths,bp,values,best,actions,history,bank,model,expander)
                go=bool(predictor(x)[0]>CFG['charge'])
                if not go:break
    return inc,go


def benchmark(output,sc,noise,test,pred,controller_refs):
    cal=np.load(output/'calibration_shared.npz');rows=[];mismatch=0;plan_error=0.
    for ci,condition in enumerate(CONDITIONS):
        name=condition['name'];model=prev.Model(condition)
        bank=Bank(cal['x'],np.load(output/f'calibration_{name}.npz')['residual'])
        expander=dict(np.load(output/f'expanders_{name}.npz'));data=np.load(output/f'data_{name}.npz')
        ref=controller_refs[ci];modes=['cb','xb',ref,'always_stop','always_continue']
        predictors={a:prev.Predictor(output/f'observer_{name}_{a}.npz') for a in ['cb','xb',ref]}
        ids=test[:CFG['runtime_cases']];totals={m:[] for m in modes}
        for mode in modes:online_case(model,bank,sc[ids[:1]],noise[:,ids[:1]],mode,predictors.get(mode),expander)
        for rep in range(CFG['runtime_repeats']):
            for mode in modes[rep:]+modes[:rep]:
                elapsed=0.
                for pos,sid in enumerate(ids):
                    t=time.perf_counter()
                    plan,go=online_case(model,bank,sc[sid:sid+1],noise[:,sid:sid+1],mode,predictors.get(mode),expander)
                    elapsed+=time.perf_counter()-t
                    expected=mode=='always_continue' if mode.startswith('always_') else bool(pred[ci,ARMS.index(mode),pos]>CFG['charge'])
                    mismatch+=int(go!=expected)
                    plan_error=max(plan_error,float(np.max(np.abs(plan-data['u6' if expected else 'u3'][sid:sid+1]))))
                totals[mode].append(elapsed*1000/len(ids))
        rows.append({'condition':name,'cases':len(ids),'repeats':CFG['runtime_repeats'],
                     'ms_per_case':{m:float(np.median(v)) for m,v in totals.items()},'repetitions_ms':totals})
        print(json.dumps({'stage':'runtime','condition':name,'ms_per_case':rows[-1]['ms_per_case']}),flush=True)
    return {'rows':rows,'decision_mismatches':mismatch,'plan_max_abs_error':plan_error,
            'runtime_saving_supported':all(row['ms_per_case']['cb']<row['ms_per_case'][controller_refs[i]] for i,row in enumerate(rows) if i>0)}


def fixtures():
    rng=np.random.default_rng(999201)
    cal=rng.uniform(-1,1,(100,4));res=np.broadcast_to([.03,-.02],(100,2)).copy();bank=Bank(cal,res)
    q=rng.uniform(-.5,.5,(8,4));pred=bank.correct(q[:,:2],q[:,2:])
    assert np.max(np.abs(pred-[.03,-.02]))<1e-10
    # Known affine residual approaches its exact value under the fixed small ridge.
    w=rng.normal(0,.01,(4,2));bank=Bank(cal,cal@w+[.03,-.02]);err=float(np.max(np.abs(bank.correct(q[:,:2],q[:,2:])-(q@w+[.03,-.02]))))
    assert err<1e-3
    costs=np.arange(32)[None,:]/10;bias=np.full_like(costs,.3);summary=bias_summary(costs,costs+bias)[0]
    assert np.allclose(summary[:5],[.3,.3,0,.3,.3],atol=1e-6)
    assert np.allclose(summary[5:8],0,atol=1e-6) and abs(summary[8]-1)<1e-6 and np.allclose(summary[9:],0)
    arrays=np.arange(20)[:,None];parts=[np.arange(12),np.arange(12,20)]
    _,mapping=prev.deranged_rows(arrays,parts,rng);assert np.all(mapping!=np.arange(20))
    j,_,go=controller(np.array([.02,-.02]),np.array([.03,-.01]),np.ones(2),.01,'cb')
    assert np.allclose(j,[.9833333333333333,1.0033333333333334]) and np.array_equal(go,[True,False])
    sc=np.array([[-.8,-.7,.8,.6,0.,0.,.2]])
    mean,sd,inc=prev.init_planner(sc);noise=rng.normal(size=(1,32,10,2));model=prev.Model(CONDITIONS[0])
    original=prev.step(model,sc,mean,sd,inc,noise);copied=step(model,sc,mean,sd,inc,noise)
    assert all(np.array_equal(a,b) for a,b in zip(original,copied[:7]))
    return {'affine_residual_max_error':err,'bias_summary':True,'strict_derangement':True,'call_charge':True,'prefix_step':True}


def run():
    output=ROOT/'run'
    if output.exists():raise RuntimeError('Preserve completed/partial runs; never silently rerun.')
    checks=fixtures();frozen=fingerprint();output.mkdir()
    write_json(output/'manifest.json',{'study':'F1-A2','version':'1.0','started_utc':datetime.now(timezone.utc).isoformat(),
               'hashes':frozen,'fixtures':checks,'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__})
    sc=scenarios();order=np.random.default_rng(CFG['split_seed']).permutation(len(sc));fit,tune,test=np.split(order,[1200,1600])
    write_npz(output/'scenarios.npz',scenarios=sc,fit=fit,tune=tune,evaluation=test)
    noise=np.random.default_rng(CFG['noise_seed']).normal(size=(6,len(sc),32,10,2))
    rng=np.random.default_rng(CFG['calibration_seed']);cal=np.c_[rng.uniform(-4,4,(2048,2)),rng.uniform(-1,1,(2048,2))]
    true_next=old.physics(cal[:,:2],cal[:,2:]);write_npz(output/'calibration_shared.npz',x=cal,true_next=true_next)
    rng=np.random.default_rng(CFG['diagnostic_seed']);ds=rng.uniform(-1.25,1.25,(256,2));da=rng.uniform(-1,1,(256,1,10,2))
    dt=old.rollout(old.physics,ds,da);write_npz(output/'diagnostic_inputs.npz',starts=ds,actions=da,true_paths=dt)
    predictions=[];tuning=[];targets=[];bases=[];refs=[];controller_refs=[];selections=[];grid=[];diagnostics=[];bias_stats=[]
    for ci,condition in enumerate(CONDITIONS):
        name=condition['name'];model=prev.Model(condition);started=time.perf_counter()
        residual=true_next-model(cal[:,:2],cal[:,2:]);bank=Bank(cal,residual);bank_seconds=time.perf_counter()-started
        write_npz(output/f'calibration_{name}.npz',residual=residual)
        original=old.rollout(model,ds,da);corrected=old.rollout(Corrected(model,bank),ds,da)
        write_npz(output/f'diagnostic_{name}.npz',model_paths=original,corrected_paths=corrected)
        data,seconds=generate(model,bank,sc,noise);write_npz(output/f'data_{name}.npz',**data)
        diagnostics.append(dict(condition,one_step_rmse=float(np.sqrt(np.mean(residual**2))),
             rollout_rmse=float(np.sqrt(np.mean((original[:,:,1:]-dt[:,:,1:])**2))),
             corrected_rollout_rmse=float(np.sqrt(np.mean((corrected[:,:,1:]-dt[:,:,1:])**2))),
             calibration_seconds=bank_seconds,generation_seconds=seconds))
        btrue=data['true_candidate_cost'][test]-data['model_cost'][test]
        bhat=data['corrected_candidate_cost'][test]-data['model_cost'][test]
        chosen=np.argmin(data['model_cost'][test],axis=1);ix=np.arange(len(test))
        bias_stats.append({'condition':name,'all_candidate_bias_mse':float(np.mean((bhat-btrue)**2)),
             'zero_bias_mse':float(np.mean(btrue**2)),
             'incumbent_bias_mse':float(np.mean((bhat[ix,chosen]-btrue[ix,chosen])**2)),
             'incumbent_zero_mse':float(np.mean(btrue[ix,chosen]**2)),
             'descriptor_rmse':np.sqrt(np.mean((data['b'][test].astype(float)-data['p'][test])**2,axis=0)).tolist(),
             'rank_discordance_mae':float(np.mean(np.abs(data['b'][test,9]-data['p'][test,9])))})
        print(json.dumps({'stage':'planning','condition':name,'seconds':seconds}),flush=True)
        matrices,expander,controls=arm_matrices(data,fit,[fit,tune,test],ci)
        write_npz(output/f'expanders_{name}.npz',**expander);write_npz(output/f'controls_{name}.npz',**controls)
        selected={};pred={};tpred={}
        for arm in ARMS:
            best,candidates=prev.fit_arm(matrices[arm],data['target'],fit,tune,output/f'observer_{name}_{arm}.npz')
            values=best.pop('predictions');pred[arm]=values[test];tpred[arm]=values[tune]
            selected[arm]=dict(best,input_dimensions=matrices[arm].shape[1])
            grid.extend([dict(condition=name,arm=arm,**v) for v in candidates])
        ref=min(REFS,key=lambda a:selected[a]['tuning_mse'])
        rc=min(REFS,key=lambda a:float(controller(tpred[a],data['target'][tune],data['base'][tune],CFG['charge'],a)[0].mean()))
        refs.append(ref);controller_refs.append(rc);selections.append(selected)
        predictions.append(np.array([pred[a] for a in ARMS]));tuning.append(np.array([tpred[a] for a in ARMS]))
        targets.append(data['target'][test]);bases.append(data['base'][test])
        print(json.dumps({'stage':'observers','condition':name,'reference':ref,'controller_reference':rc}),flush=True)
    pred=np.array(predictions);y=np.array(targets);base=np.array(bases)
    write_npz(output/'evaluation.npz',predictions=pred,tuning_predictions=np.array(tuning),target=y,base=base,scenario_ids=test)
    write_json(output/'candidate_grid.json',grid)
    result=analyze(pred,y,base,refs,controller_refs)
    result.update(study='F1-A2',version='1.0',conditions=CONDITIONS,arms=ARMS,bias_names=B_NAMES,
                  reference_arms=REFS,references=refs,controller_references=controller_refs,
                  selected_observers=selections,candidate_fits=len(grid),selected_pipelines=len(ARMS)*len(CONDITIONS),
                  model_diagnostics=diagnostics,bias_estimation=bias_stats)
    assert len(grid)==1260
    write_json(output/'result.json',result)
    print(json.dumps({'stage':'headlines','headline':result['headline'],'verdicts':result['verdicts']}),flush=True)
    runtime=benchmark(output,sc,noise,test,pred,controller_refs);write_json(output/'runtime.json',runtime)
    assert runtime['decision_mismatches']==0 and runtime['plan_max_abs_error']<1e-9
    assert fingerprint()==frozen
    result['runtime']=runtime;result['finished_utc']=datetime.now(timezone.utc).isoformat();write_json(output/'result.json',result)
    print(json.dumps({'stage':'finished','verdicts':result['verdicts']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fixtures-only',action='store_true');args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures_only:print(json.dumps(fixtures()))
        else:run()
