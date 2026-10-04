"""Independent arithmetic/reconstruction audit; no fitting or selection changes."""
import json, time
import numpy as np
from threadpoolctl import threadpool_limits
import study as s

def true_cost(sc,actions):
    if actions.ndim==3:actions=actions[:,None]
    position=np.broadcast_to(sc[:,None,:2],actions.shape[:2]+(2,)).copy()
    goal_sum=np.zeros(actions.shape[:2]); obstacle_sum=goal_sum.copy()
    for k in range(actions.shape[2]):
        x=position[...,0].copy(); y=position[...,1].copy(); a=actions[:,:,k]
        position=np.stack([x+.25*(a[...,0]+.25*np.sin(1.8*y)+.08*x),
                           y+.25*(a[...,1]+.2*np.sin(1.5*x)-.1*y)],axis=-1)
        distance2=np.sum((position-sc[:,None,2:4])**2,axis=-1); goal_sum+=distance2
        distance=np.sqrt(np.sum((position-sc[:,None,4:6])**2,axis=-1))
        obstacle_sum+=np.maximum(1-distance/sc[:,None,6],0)**2
    return distance2+.03*goal_sum/actions.shape[2]+12*obstacle_sum/actions.shape[2]+.01*np.mean(np.sum(actions**2,axis=-1),axis=-1)

def run():
    started=time.perf_counter(); out=s.ROOT/'run'
    manifest=json.loads((out/'manifest.json').read_text()); result=json.loads((out/'result.json').read_text())
    assert s.fingerprint()==manifest['hashes'] and (out/'COMPLETE.json').exists()
    q=np.load(out/'scenarios.npz'); sc=q['scenarios']; fit,tune,test=[q[k] for k in ['fit','tune','evaluation']]
    assert [len(fit),len(tune),len(test)]==[1200,400,800]
    assert len(set(fit)|set(tune)|set(test))==2400 and not(set(fit)&set(tune) or set(fit)&set(test) or set(tune)&set(test))
    assert np.array_equal(sc,s.scenes())
    ev=np.load(out/'evaluation.npz'); grid=json.loads((out/'candidate_grid.json').read_text())
    noise=np.random.default_rng(s.CFG['noise_seed']).normal(size=(3,len(sc),32,10,2))
    errors={k:0. for k in ['true_cost','prefix','subset_rollout','normalization','grid_score','prediction','headline']}
    blocked=0; counted=0; selfpairs=0
    for ci,condition in enumerate(s.CONDITIONS):
        name=condition['name']; d=dict(np.load(out/f'data_{name}.npz'))
        for key,acts in [('base','u3'),('final','u6'),('true_candidate_cost','actions3')]:
            actual=true_cost(sc,d[acts]); actual=actual[:,0] if key!='true_candidate_cost' else actual
            errors['true_cost']=max(errors['true_cost'],float(np.max(np.abs(actual-d[key]))))
        assert np.array_equal(d['target'],d['base']-d['final']) and np.array_equal(ev['target'][ci],d['target'][test])
        assert np.array_equal(s.subset(d['model_cost']),d['subset_ids'])
        assert np.all(d['subset_ids'][:,0]==np.argmin(d['model_cost'],axis=1))
        assert all(len(set(row))==8 for row in d['subset_ids'])
        model=s.p.Model(condition); bank=s.loadbank(condition)
        n=100; scene=sc[:n]; mean,sd,inc=s.p.init_planner(scene); history=[]
        for k in range(3):
            mean,sd,inc,paths,bp,values,best,actions=s.a2.step(model,scene,mean,sd,inc,noise[k,:n])
            if k<2:history.append(s.p.current(scene,mean,sd,inc,values,best))
        current=s.p.current(scene,mean,sd,inc,values,best)
        feats=s.p.features(scene,current,np.stack(history,axis=1),paths,bp,inc,bank)
        for key,value in feats.items():errors['prefix']=max(errors['prefix'],float(np.max(np.abs(value.astype(float)-d[key][:n]))))
        assert np.array_equal(inc,d['u3'][:n]) and np.array_equal(actions,d['actions3'][:n])
        physics=s.old.physics
        if condition['alpha']==1:
            def forbidden(*args,**kwargs):raise AssertionError('Accessible subset called the true simulator')
            s.old.physics=forbidden; blocked+=1
        try:
            ids=s.subset(values); a=actions[np.arange(n)[:,None],ids]
            tc=s.old.cost(s.old.rollout(s.a2.Corrected(model,bank),scene[:,:2],a),a,scene)
        finally:s.old.physics=physics
        expected=np.take_along_axis(d['corrected_candidate_cost'][:n],ids,axis=1)
        errors['subset_rollout']=max(errors['subset_rollout'],float(np.max(np.abs(tc-expected))))
        b8=s.summary8(np.take_along_axis(values,ids,1),tc)
        errors['prefix']=max(errors['prefix'],float(np.max(np.abs(b8-d['b8'][:n]))))
        mats,expander,source=s.matrices(d,fit,[fit,tune,test],ci)
        saved=np.load(out/f'controls_{name}.npz')['source']; assert np.array_equal(source,saved)
        selfpairs+=int(np.sum(source==np.arange(len(sc))))
        for ix in [fit,tune,test]:assert set(source[ix])==set(ix)
        # Features must not depend on labels, future plans or privileged costs.
        forbidden_fields=['target','base','final','u6','p','true_candidate_cost','estimated_costs']
        poisoned=dict(d)
        for key in forbidden_fields:poisoned[key]=np.full_like(d[key],np.nan)
        poisoned_mats,_,_=s.matrices(poisoned,fit,[fit,tune,test],ci)
        assert all(np.array_equal(mats[a],poisoned_mats[a]) for a in s.ARMS)
        scores={}; tp={}
        for ai,arm in enumerate(s.ARMS):
            x=mats[arm]; state=dict(np.load(out/f'observer_{name}_{arm}.npz'))
            _,xm,xs=s.p.standardize(x,fit)
            errors['normalization']=max(errors['normalization'],float(np.max(np.abs(xm-state['xm']))),float(np.max(np.abs(xs-state['xs']))))
            assert np.isclose(state['ym'],d['target'][fit].mean())
            z=(x-state['xm'])/state['xs']; rows=[r for r in grid if r['condition']==name and r['arm']==arm]
            assert len(rows)==12
            for bi in range(3):
                basis=s.p.basis(z,bi); _,bm,bs=s.p.standardize(basis,fit)
                errors['normalization']=max(errors['normalization'],float(np.max(np.abs(bm-state[f'b{bi}_mean']))),float(np.max(np.abs(bs-state[f'b{bi}_sd']))))
                norm=(basis-state[f'b{bi}_mean'])/state[f'b{bi}_sd']
                for aj in range(4):
                    pred=norm@state[f'b{bi}_beta'][:,aj]+state['ym']; mse=float(np.mean((pred[tune]-d['target'][tune])**2))
                    row=next(r for r in rows if r['basis_index']==bi and r['alpha_index']==aj)
                    errors['grid_score']=max(errors['grid_score'],abs(mse-row['tuning_mse'])); counted+=1
            chosen=min(rows,key=lambda r:r['tuning_mse'])
            assert chosen['basis_index']==int(state['selected_basis']) and chosen['alpha_index']==int(state['selected_alpha'])
            pred=s.p.Predictor(out/f'observer_{name}_{arm}.npz')(x)
            errors['prediction']=max(errors['prediction'],float(np.max(np.abs(pred[test]-ev['predictions'][ci,ai]))),float(np.max(np.abs(pred[tune]-ev['tuning_predictions'][ci,ai]))))
            tp[arm]=pred[tune]; scores[arm]=chosen['tuning_mse']
        assert min(s.REFS,key=lambda a:scores[a])==result['references'][ci]
        ref=min(s.REFS,key=lambda a:float(s.controller(tp[a],d['target'][tune],d['base'][tune],a)[0].mean()))
        assert ref==result['controller_references'][ci]
        timing=np.load(out/f'runtime_{name}.npz'); rt=json.loads((out/f'runtime_{name}.json').read_text())
        for mi,arm in enumerate(rt['modes']):
            go=np.full(s.CFG['runtime_cases'],arm=='always_continue') if arm.startswith('always_') else ev['predictions'][ci,s.ARMS.index(arm),:s.CFG['runtime_cases']]>.01
            expected=960+960*go+s.extra(arm)
            assert np.array_equal(timing['model_transitions'][mi],np.tile(expected,(s.CFG['runtime_repeats'],1)))
            ms=float(np.median(timing['milliseconds'][mi],axis=0).mean())
            assert abs(ms-rt['ms_per_case'][arm])<1e-12
        print(json.dumps({'stage':'audited_condition','condition':name}),flush=True)
    assert counted==1008 and selfpairs==0 and blocked==3
    pred=ev['predictions']; y=ev['target']; base=ev['base']
    b8=pred[:,s.ARMS.index('b8')]; b32=pred[:,s.ARMS.index('b32')]
    ref=np.array([pred[i,s.ARMS.index(a)] for i,a in enumerate(result['references'])])
    rc=np.array([pred[i,s.ARMS.index(a)] for i,a in enumerate(result['controller_references'])])
    er=(ref-y)**2; e8=(b8-y)**2; e32=(b32-y)**2
    jr=base-np.where(rc>.01,y-.01,0); jc=base-np.where(b8>.01,y-.01,0)+.01/12
    boot=np.random.default_rng(s.CFG['bootstrap_seed']).integers(0,800,(4000,800))
    for key,v in [('useful_signal',(er-e8)[1:].mean(0)),('charged_controller',(jr-jc)[1:].mean(0))]:
        ci=np.quantile(v[boot].mean(1),[.05/6,1-.05/6]); saved=result['headline'][key]
        errors['headline']=max(errors['headline'],abs(v.mean()-saved['mean']),float(np.max(np.abs(ci-saved['ci_adjusted']))))
    d=(e8-e32)[1:].mean(0); den=e32[1:].mean(0)
    ci=np.quantile(d[boot].mean(1)/den[boot].mean(1),[.05/6,1-.05/6])
    errors['headline']=max(errors['headline'],float(np.max(np.abs(ci-result['headline']['accuracy_retained']['ci_adjusted']))))
    assert max(errors.values())<1e-9 and s.fingerprint()==manifest['hashes']
    rt=result['runtime']; assert rt['decision_mismatches']==0 and rt['plan_max_abs_error']<1e-9
    verdicts=result['primary_verdicts']; assert result['practical_success']==(all(verdicts.values()) and rt['runtime_gate'])
    audit={'passed':True,'candidate_tuning_scores_reconstructed':counted,'selected_pipelines_reconstructed':84,
        'max_absolute_errors':errors,'prefix_cases_per_condition':100,'blocked_true_simulator_models':blocked,
        'future_and_privileged_field_poisoning_passed':True,'mismatch_self_pairs':selfpairs,
        'reference_selections_verified':True,'online_model_transition_counts_verified':True,
        'runtime_cases_verified':7*5*32*5,'source_hashes_unchanged':True,'additional_fits':0,
        'seconds':time.perf_counter()-started}
    audit_path=out/'audit.json'
    if audit_path.exists():audit_path=out/'audit_recheck.json'
    s.wj(audit_path,audit); print(json.dumps(audit,indent=2))

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
