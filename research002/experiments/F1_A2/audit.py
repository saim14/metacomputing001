"""Post-run reconstruction, no additional fitting or candidate selection."""
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
import study as s


def independent_cost(sc,actions):
    if actions.ndim==3:actions=actions[:,None]
    position=np.broadcast_to(sc[:,None,:2],actions.shape[:2]+(2,)).copy()
    goal_sum=np.zeros(actions.shape[:2]);obstacle_sum=goal_sum.copy()
    for k in range(actions.shape[2]):
        x=position[...,0].copy();y=position[...,1].copy();a=actions[:,:,k]
        position=np.stack([x+.25*(a[...,0]+.25*np.sin(1.8*y)+.08*x),
                           y+.25*(a[...,1]+.2*np.sin(1.5*x)-.1*y)],axis=-1)
        distance2=np.sum((position-sc[:,None,2:4])**2,axis=-1);goal_sum+=distance2
        distance=np.sqrt(np.sum((position-sc[:,None,4:6])**2,axis=-1))
        obstacle_sum+=np.maximum(1-distance/sc[:,None,6],0)**2
    return distance2+.03*goal_sum/actions.shape[2]+12*obstacle_sum/actions.shape[2]+.01*np.mean(np.sum(actions**2,axis=-1),axis=-1)


def run():
    out=s.ROOT/'run';started=time.perf_counter()
    result=json.loads((out/'result.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    assert s.fingerprint()==manifest['hashes']
    scenes=np.load(out/'scenarios.npz');sc=scenes['scenarios'];fit,tune,test=[scenes[k] for k in ['fit','tune','evaluation']]
    assert [len(fit),len(tune),len(test)]==[1200,400,800]
    assert len(set(fit)|set(tune)|set(test))==2400 and not(set(fit)&set(tune) or set(fit)&set(test) or set(tune)&set(test))
    ev=np.load(out/'evaluation.npz');grid=json.loads((out/'candidate_grid.json').read_text());cal=np.load(out/'calibration_shared.npz')
    noise=np.random.default_rng(s.CFG['noise_seed']).normal(size=(3,len(sc),32,10,2))
    errors={k:0. for k in ['true_outcome','true_candidate_cost','prefix_feature','normalization','candidate_tuning_score','selected_prediction','headline']}
    counted=0;self_pairs=0;blocked_models=0
    for ci,condition in enumerate(s.CONDITIONS):
        name=condition['name'];data=dict(np.load(out/f'data_{name}.npz'))
        for key,ukey in [('base','u3'),('final','u6')]:
            error=float(np.max(np.abs(independent_cost(sc,data[ukey])[:,0]-data[key])))
            errors['true_outcome']=max(errors['true_outcome'],error)
        errors['true_candidate_cost']=max(errors['true_candidate_cost'],float(np.max(np.abs(independent_cost(sc,data['actions3'])-data['true_candidate_cost']))))
        assert np.array_equal(data['target'],data['base']-data['final']) and np.array_equal(ev['target'][ci],data['target'][test])
        model=s.prev.Model(condition);residual=np.load(out/f'calibration_{name}.npz')['residual'];bank=s.Bank(cal['x'],residual)
        assert np.allclose(residual,cal['true_next']-model(cal['x'][:,:2],cal['x'][:,2:]),rtol=0,atol=1e-12)
        n=100;scene=sc[:n];mean,sd,inc=s.prev.init_planner(scene);history=[]
        for k in range(3):
            mean,sd,inc,paths,bp,values,best,actions=s.step(model,scene,mean,sd,inc,noise[k,:n])
            c=s.prev.current(scene,mean,sd,inc,values,best)
            if k<2:history.append(c.copy())
        feature=s.prev.features(scene,c,np.stack(history,axis=1),paths,bp,inc,bank)
        tc=s.old.cost(s.old.rollout(s.old.physics,scene[:,:2],actions),actions,scene)
        # Block the true simulator while computing accessible B for full neural models.
        physics=s.old.physics
        if condition['alpha']==1:
            def forbidden(*args,**kwargs):raise AssertionError('Accessible B called the true simulator')
            s.old.physics=forbidden;blocked_models+=1
        try:bc=s.old.cost(s.old.rollout(s.Corrected(model,bank),scene[:,:2],actions),actions,scene)
        finally:s.old.physics=physics
        feature.update(p=s.bias_summary(values,tc),b=s.bias_summary(values,bc))
        for key,value in feature.items():
            errors['prefix_feature']=max(errors['prefix_feature'],float(np.max(np.abs(value.astype(float)-data[key][:n].astype(float)))))
        assert np.array_equal(inc,data['u3'][:n]) and np.array_equal(actions,data['actions3'][:n])
        mats,expander,controls=s.arm_matrices(data,fit,[fit,tune,test],ci)
        saved=np.load(out/f'controls_{name}.npz')
        for key,mapping in controls.items():
            assert np.array_equal(mapping,saved[key]);self_pairs+=int(np.sum(mapping==np.arange(len(sc))))
            for ids in [fit,tune,test]:assert set(mapping[ids])==set(ids)
        tp={};tune_scores={}
        for ai,arm in enumerate(s.ARMS):
            x=mats[arm];state=dict(np.load(out/f'observer_{name}_{arm}.npz'))
            _,xm,xs=s.standardize(x,fit)
            errors['normalization']=max(errors['normalization'],float(np.max(np.abs(xm-state['xm']))),float(np.max(np.abs(xs-state['xs']))))
            z=(x-state['xm'])/state['xs'];records=[r for r in grid if r['condition']==name and r['arm']==arm]
            assert len(records)==12
            for bi in range(3):
                b=s.prev.basis(z,bi);_,bm,bs=s.standardize(b,fit)
                errors['normalization']=max(errors['normalization'],float(np.max(np.abs(bm-state[f'b{bi}_mean']))),float(np.max(np.abs(bs-state[f'b{bi}_sd']))))
                normalized=(b-state[f'b{bi}_mean'])/state[f'b{bi}_sd']
                for aj in range(4):
                    pred=normalized@state[f'b{bi}_beta'][:,aj]+state['ym'];mse=float(np.mean((pred[tune]-data['target'][tune])**2))
                    stored=next(r for r in records if r['basis_index']==bi and r['alpha_index']==aj)
                    errors['candidate_tuning_score']=max(errors['candidate_tuning_score'],abs(mse-stored['tuning_mse']));counted+=1
            selected=min(records,key=lambda r:r['tuning_mse'])
            assert selected['basis_index']==int(state['selected_basis']) and selected['alpha_index']==int(state['selected_alpha'])
            pred=s.prev.Predictor(out/f'observer_{name}_{arm}.npz')(x)
            errors['selected_prediction']=max(errors['selected_prediction'],float(np.max(np.abs(pred[test]-ev['predictions'][ci,ai]))),float(np.max(np.abs(pred[tune]-ev['tuning_predictions'][ci,ai]))))
            tp[arm]=pred[tune];tune_scores[arm]=selected['tuning_mse']
        assert min(s.REFS,key=lambda a:tune_scores[a])==result['references'][ci]
        rc=min(s.REFS,key=lambda a:float(s.controller(tp[a],data['target'][tune],data['base'][tune],.01,a)[0].mean()))
        assert rc==result['controller_references'][ci]
    assert counted==1260 and self_pairs==0 and blocked_models==3
    p=ev['predictions'];y=ev['target'];base=ev['base']
    ref=np.array([p[i,s.ARMS.index(a)] for i,a in enumerate(result['references'])]);cp=p[:,s.ARMS.index('cp')];cb=p[:,s.ARMS.index('cb')]
    pc=np.array([p[i,s.ARMS.index(a)] for i,a in enumerate(result['controller_references'])])
    jref=base-np.where(pc>.01,y-.01,0);jcb=base-np.where(cb>.01,y-.01,0)+.01/3
    values={'privileged_signal':((ref-y)**2-(cp-y)**2)[1:].mean(0),
            'estimated_signal':((ref-y)**2-(cb-y)**2)[1:].mean(0),'charged_controller':(jref-jcb)[1:].mean(0)}
    boot=np.random.default_rng(66001).integers(0,800,(4000,800))
    for key,v in values.items():
        actual=np.quantile(v[boot].mean(1),[.05/6,1-.05/6]);stored=result['headline'][key]
        errors['headline']=max(errors['headline'],abs(float(v.mean())-stored['mean']),float(np.max(np.abs(actual-stored['ci_adjusted']))))
    assert max(errors.values())<1e-9
    assert result['runtime']['decision_mismatches']==0 and result['runtime']['plan_max_abs_error']<1e-9
    assert s.fingerprint()==manifest['hashes']
    audit={'passed':True,'candidate_tuning_scores_reconstructed':counted,'selected_pipelines_reconstructed':105,
           'max_absolute_errors':errors,'prefix_cases_per_condition':100,'mismatch_self_pairs':self_pairs,
           'full_models_audited_with_true_simulator_blocked':blocked_models,
           'all_reference_selections_verified':True,'online_replay_verified':True,
           'source_and_checkpoint_hashes_unchanged':True,'additional_observer_fits':0,'additional_neural_fits':0,
           'audit_seconds':time.perf_counter()-started}
    s.write_json(out/'audit.json',audit);print(json.dumps(audit,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
