"""Reconstruct F1-A1 from saved coefficients and data without fitting."""
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
import study as s


def execute_true(sc,actions):
    position=sc[:,:2].copy(); goal_cost=[]; obstacle_cost=[]
    for action in actions.transpose(1,0,2):
        x,y=position.T.copy()
        position=np.column_stack([x+.25*(action[:,0]+.25*np.sin(1.8*y)+.08*x),
                                  y+.25*(action[:,1]+.20*np.sin(1.5*x)-.10*y)])
        goal_cost.append(np.sum((position-sc[:,2:4])**2,axis=1))
        distance=np.sqrt(np.sum((position-sc[:,4:6])**2,axis=1))
        obstacle_cost.append(np.maximum(1-distance/sc[:,6],0)**2)
    return goal_cost[-1]+.03*np.mean(goal_cost,axis=0)+12*np.mean(obstacle_cost,axis=0)+.01*np.mean(np.sum(actions**2,axis=2),axis=1)


def main():
    output=s.ROOT/'run'; started=time.perf_counter()
    result=json.loads((output/'result.json').read_text())
    manifest=json.loads((output/'manifest.json').read_text())
    assert s.fingerprint()==manifest['hashes']
    scene=np.load(output/'scenarios.npz'); sc=scene['scenarios']
    fit,tune,test=[scene[k] for k in ['fit','tune','evaluation']]
    assert len(set(fit)|set(tune)|set(test))==2400 and len(test)==800
    assert not(set(fit)&set(tune) or set(fit)&set(test) or set(tune)&set(test))
    ev=np.load(output/'evaluation.npz'); grid=json.loads((output/'candidate_grid.json').read_text())
    cal=np.load(output/'calibration_shared.npz')
    errors={'candidate_tuning_score':0.,'selected_prediction':0.,'true_outcome':0.,'prefix_feature':0.,'fit_normalization':0.}
    checked=0; mismatches=0; historical_swaps=[]
    noise=np.random.default_rng(53001).normal(size=(s.CFG['checkpoint'],len(sc),s.CFG['candidates'],s.CFG['horizon'],2))
    for ci,condition in enumerate(s.CONDITIONS):
        name=condition['name']; data=dict(np.load(output/f'data_{name}.npz'))
        for key,actions in [('base',data['u3']),('final',data['u6'])]:
            actual=execute_true(sc,actions)
            errors['true_outcome']=max(errors['true_outcome'],float(np.max(np.abs(actual-data[key]))))
        assert np.max(np.abs(data['target']-(data['base']-data['final'])))==0
        assert np.array_equal(data['target'][test],ev['target'][ci])
        matrices,expander,controls=s.arm_matrices(data,fit,[fit,tune,test],ci)
        saved_controls=np.load(output/f'controls_{name}.npz')
        for key in controls:assert np.array_equal(controls[key],saved_controls[key])
        for key in ['h_source','g_source','r_source']:
            mapping=controls[key]; mismatches+=int(np.sum(mapping==np.arange(len(sc))))
            for ids in [fit,tune,test]:assert set(mapping[ids])==set(ids)
        historical_swaps.append(float(controls['past_swapped'].mean()))
        # Rebuild round-3 observables with a prefix replay only.
        model=s.Model(condition)
        bank=s.Reliability(cal['x'],np.load(output/f'calibration_{name}.npz')['error2'])
        sub=sc[:100]; mean,sd,inc=s.init_planner(sub); history=[]
        for k in range(3):
            mean,sd,inc,paths,bp,values,best=s.step(model,sub,mean,sd,inc,noise[k,:100])
            c=s.current(sub,mean,sd,inc,values,best)
            if k<2:history.append(c.copy())
        feature=s.features(sub,c,np.stack(history,axis=1),paths,bp,inc,bank)
        for key in ['c','h','g','r','raw']:
            errors['prefix_feature']=max(errors['prefix_feature'],float(np.max(np.abs(feature[key].astype(float)-data[key][:100].astype(float)))))
        selected_tuning={}
        for ai,arm in enumerate(s.ARMS):
            x=matrices[arm]; state=dict(np.load(output/f'observer_{name}_{arm}.npz'))
            _,xm,xs=s.standardize(x,fit)
            errors['fit_normalization']=max(errors['fit_normalization'],float(np.max(np.abs(xm-state['xm']))),float(np.max(np.abs(xs-state['xs']))))
            z=(x-state['xm'])/state['xs']
            candidates=[row for row in grid if row['condition']==name and row['arm']==arm]
            assert len(candidates)==12
            for bi in range(3):
                b=s.basis(z,bi)
                _,bm,bs=s.standardize(b,fit)
                errors['fit_normalization']=max(errors['fit_normalization'],float(np.max(np.abs(bm-state[f'b{bi}_mean']))),float(np.max(np.abs(bs-state[f'b{bi}_sd']))))
                norm=(b-state[f'b{bi}_mean'])/state[f'b{bi}_sd']
                for penalty_idx in range(4):
                    # Same vector-dot operation as the frozen fit, no solve/refit.
                    p=norm@state[f'b{bi}_beta'][:,penalty_idx]+state['ym']
                    mse=float(np.mean((p[tune]-data['target'][tune])**2))
                    row=next(v for v in candidates if v['basis_index']==bi and v['alpha_index']==penalty_idx)
                    errors['candidate_tuning_score']=max(errors['candidate_tuning_score'],abs(mse-row['tuning_mse']))
                    checked+=1
            chosen=min(candidates,key=lambda v:v['tuning_mse'])
            assert chosen['basis_index']==int(state['selected_basis']) and chosen['alpha_index']==int(state['selected_alpha'])
            pred=s.Predictor(output/f'observer_{name}_{arm}.npz')(x)
            errors['selected_prediction']=max(errors['selected_prediction'],float(np.max(np.abs(pred[test]-ev['predictions'][ci,ai]))))
            assert np.allclose(pred[tune],ev['tuning_predictions'][ci,ai],rtol=0,atol=1e-10)
            selected_tuning[arm]=pred[tune]
        ref0=min(s.REF0,key=lambda arm:result['selected_observers'][ci][arm]['tuning_mse'])
        ref_r=min(s.REFR,key=lambda arm:result['selected_observers'][ci][arm]['tuning_mse'])
        ref_c=min(s.REFR,key=lambda arm:float(s.controller(selected_tuning[arm],data['target'][tune],data['base'][tune],.01)[0].mean()))
        assert ref0==result['primary_refs'][ci] and ref_r==result['reliability_refs'][ci] and ref_c==result['controller_refs'][ci]
    assert checked==1344 and mismatches==0
    assert max(errors.values())<1e-9
    # Independent four-endpoint construction and paired percentile intervals.
    p=ev['predictions']; y=ev['target']; base=ev['base']
    full=[i for i,c in enumerate(s.CONDITIONS) if c['alpha']==1]
    zero_harm=(y[0]<-.01).astype(float)
    intervention=np.mean((y[full]<-.01).astype(float)-zero_harm,axis=0)
    ref=np.array([p[i,s.ARMS.index(a)] for i,a in enumerate(result['primary_refs'])])
    hg=p[:,s.ARMS.index('chg')]; hgr=p[:,s.ARMS.index('chgr')]
    joint=np.mean(((ref-y)**2-(hg-y)**2)[1:],axis=0)
    reliability=np.mean(((hg-y)**2-(hgr-y)**2)[1:],axis=0)
    reference=np.array([p[i,s.ARMS.index(a)] for i,a in enumerate(result['controller_refs'])])
    j_ref=base-np.where(reference>.01,y-.01,0)
    j_candidate=base-np.where(hgr>.01,y-.01,0)
    policy=np.mean((j_ref-j_candidate)[1:],axis=0)
    boot=np.random.default_rng(56001).integers(0,800,(4000,800))
    max_interval=0.
    for key,v in [('error_intervention',intervention),('joint_history_geometry',joint),('added_reliability',reliability),('charged_controller',policy)]:
        samples=v[boot].mean(1)
        actual=np.quantile(samples,[.00625,.99375])
        stored=result['headline'][key]
        max_interval=max(max_interval,float(np.max(np.abs(actual-stored['ci9875']))),abs(float(v.mean())-stored['mean']))
    assert max_interval<1e-12
    assert result['runtime']['decision_mismatches']==0 and result['runtime']['plan_max_abs_error']<1e-9
    assert s.fingerprint()==manifest['hashes']
    audit={'passed':True,'candidate_tuning_scores_reconstructed':checked,'selected_pipelines_reconstructed':112,
           'max_absolute_errors':errors,'headline_interval_max_abs_error':max_interval,
           'mismatch_self_pairs':mismatches,'past_swap_fractions':historical_swaps,
           'prefix_cases_per_condition':100,'source_and_checkpoint_hashes_unchanged':True,
           'all_reference_selections_verified':True,'online_decisions_verified':True,
           'additional_neural_fits':0,'additional_observer_fits':0,'audit_seconds':time.perf_counter()-started}
    s.write_json(output/'audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
