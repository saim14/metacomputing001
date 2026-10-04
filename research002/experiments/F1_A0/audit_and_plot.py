"""Read-only numerical reconstruction of F1-A0; no new model or observer fits."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from run_f1 import ROOT, CFG, ARMS, World, arms, geometry, rollout, cost, ci, allocated


def replay_true(sc, actions):
    # Independent expression of the simulator and cost; actions are fixed records.
    pos=sc[:,:2].copy()
    squared_dist=[]; penetration=[]
    for action in actions.transpose(1,0,2):
        x,y=pos.T.copy()
        pos=np.column_stack([x+.25*(action[:,0]+.25*np.sin(1.8*y)+.08*x),
                             y+.25*(action[:,1]+.20*np.sin(1.5*x)-.10*y)])
        squared_dist.append(np.square(pos-sc[:,2:4]).sum(axis=1))
        distance=np.sqrt(np.square(pos-sc[:,4:6]).sum(axis=1))
        penetration.append(np.square(np.maximum(1-distance/sc[:,6],0)))
    return squared_dist[-1]+.03*np.mean(squared_dist,axis=0)+12*np.mean(penetration,axis=0)+.01*np.mean(np.square(actions).sum(axis=2),axis=1)


def predict(x, path):
    with np.load(path) as f:
        state=dict(f)
    z=(x-state['xm'])/state['xs']
    if state['w'].size:
        z=np.sqrt(2/CFG['rff_width'])*np.cos(z@state['w']+state['phase'])
    return ((z-state['bm'])/state['bs'])@state['beta']+state['ym']


def main():
    output=ROOT/'run'
    result=json.loads((output/'result.json').read_text())
    manifest=json.loads((output/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    d=np.load(output/'scenarios.npz')
    sc,fit,tune,test=[d[k] for k in ['scenarios','fit','tune','evaluation']]
    ev=np.load(output/'evaluation.npz')
    grid=json.loads((output/'candidate_grid.json').read_text())
    max_prediction=max_tuning=max_target=max_prefix=0.
    fixed_points=[]; inference=[]; all_error=[]
    # Prefix random draws match the frozen generation stream; only two rounds run.
    noise=np.random.default_rng(43001).normal(size=(2,len(sc),CFG['candidates'],CFG['horizon'],2))
    for mi,seed in enumerate(CFG['model_seeds']):
        data=dict(np.load(output/f'planning_{seed}.npz'))
        matrices=arms(data,fit,[fit,tune,test],mi)
        base=replay_true(sc,data['u2']); final=replay_true(sc,data['u6'])
        target_error=float(np.max(np.abs(base-final-data['target'])))
        max_target=max(max_target,target_error)
        assert target_error<1e-10
        counts={}
        rng=np.random.default_rng(44002+mi)
        for name,idx in [('fit',fit),('tuning',tune),('evaluation',test)]:
            counts[name]=int(np.sum(rng.permutation(idx)==idx))
        fixed_points.append({'seed':seed,**counts})
        with np.load(output/f'world_{seed}.npz') as w:
            world=World(dict(w))
        # All 100 records in the first planning batch are reproduced without
        # executing rounds 3-6, demonstrating the saved inputs need no continuation.
        subset=sc[:100]; m=len(subset); h=CFG['horizon']
        mean=np.repeat(np.clip((subset[:,2:4]-subset[:,:2])/(.25*h),-.8,.8)[:,None,:],h,axis=1)
        sd=np.full_like(mean,.65); incumbent=mean.copy()
        for k in range(2):
            a=np.clip(mean[:,None]+sd[:,None]*noise[k,:100],-1,1)
            a[:,0],a[:,1]=incumbent,mean
            paths=rollout(world,subset[:,:2],a)
            values=cost(paths,a,subset)
            order=np.argsort(values,axis=1,kind='stable')
            incumbent=a[np.arange(m),order[:,0]].copy()
            best=values[np.arange(m),order[:,0]]
            elite=a[np.arange(m)[:,None],order[:,:CFG['elites']]]
            mean=elite.mean(1); sd=np.maximum(elite.std(1),.08)
        current=np.c_[subset,mean.reshape(m,-1),sd.reshape(m,-1),incumbent.reshape(m,-1),np.sort(values,axis=1),best]
        prefix_error=max(float(np.max(np.abs(current-data['current'][:100]))),
                         float(np.max(np.abs(geometry(paths,subset)-data['geometry'][:100]))),
                         float(np.max(np.abs(paths.reshape(m,-1)-data['raw'][:100]))))
        max_prefix=max(max_prefix,prefix_error)
        assert prefix_error<1e-10
        timed={}
        for arm in ARMS:
            p=output/f'observer_{seed}_{arm}.npz'
            pred=predict(matrices[arm],p)
            err=float(np.max(np.abs(pred[test]-ev[arm][mi])))
            max_prediction=max(max_prediction,err)
            selected=result['selected_observers'][mi][arm]
            actual_mse=float(np.mean((pred[tune]-data['target'][tune])**2))
            max_tuning=max(max_tuning,abs(actual_mse-selected['tuning_mse']))
            candidates=[r for r in grid if r['seed']==seed and r['arm']==arm]
            chosen=min(candidates,key=lambda row:row['tuning_mse'])
            assert chosen['kind']==selected['kind'] and chosen['alpha']==selected['alpha']
            assert err<1e-10 and abs(actual_mse-selected['tuning_mse'])<1e-10
            # Include transforms / coefficients, exclude disk loading and feature acquisition.
            state=dict(np.load(p)); x=matrices[arm][test]
            def inference_once():
                z=(x-state['xm'])/state['xs']
                if state['w'].size:z=np.sqrt(2/CFG['rff_width'])*np.cos(z@state['w']+state['phase'])
                return ((z-state['bm'])/state['bs'])@state['beta']+state['ym']
            for _ in range(2):inference_once()
            elapsed=[]
            for _ in range(7):
                started=time.perf_counter(); inference_once(); elapsed.append(time.perf_counter()-started)
            timed[arm]=float(np.median(elapsed)*1000)
        inference.append({'seed':seed,'batch_size':len(test),'prepared_feature_inference_ms':timed})
    boot=np.random.default_rng(46001).integers(0,len(test),(CFG['bootstrap'],len(test)))
    er=(ev['reference']-ev['target'])**2; eg=(ev['geometry']-ev['target'])**2
    reconstructed=ci(er-eg,boot)
    assert reconstructed['mean']==result['primary_mse_difference']['mean']
    assert reconstructed['ci95']==result['primary_mse_difference']['ci95']
    lg,mg=allocated(ev['geometry'],ev['target'],ev['base'],.5)
    lr,mr=allocated(ev['reference'],ev['target'],ev['base'],.5)
    assert np.all(mg.sum(1)==150) and np.all(mr.sum(1)==150)
    assert ci(lr-lg,boot)==result['policies']['0.5']['benefit_vs_reference']
    assert result['status']=='combined_advancement_criterion_not_met' and not any(result['criteria'].values())
    audit={'passed':True,'selected_pipelines_reconstructed':15,'prediction_max_abs_error':max_prediction,
           'selected_tuning_mse_max_abs_error':max_tuning,'true_value_replay_max_abs_error':max_target,
           'two_round_prefix_max_abs_error':max_prefix,'prefix_scenarios_per_model':100,
           'source_hashes_unchanged':True,'bootstrap_and_allocation_reconstructed':True,
           'mismatch_permutation_fixed_points':fixed_points,'inference_timings':inference,
           'additional_model_fits':0,'additional_observer_fits':0}
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    # Static research figure: uncertainty resamples scenarios jointly, fixed models.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fafbf8','axes.facecolor':'#fafbf8'})
    fig,ax=plt.subplots(1,2,figsize=(13,5.4),gridspec_kw={'width_ratios':[1.05,1]})
    estimates=[]; bounds=[]
    for i in range(4):
        ref=er[i] if i<3 else er.mean(0)
        geo=eg[i] if i<3 else eg.mean(0)
        estimates.append(100*(ref.mean()-geo.mean())/ref.mean())
        samples=100*(ref[boot].mean(1)-geo[boot].mean(1))/ref[boot].mean(1)
        bounds.append(np.quantile(samples,[.025,.975]))
    estimates=np.array(estimates); bounds=np.array(bounds)
    for i in range(4):
        color='#295d4b' if i==3 else '#718d80'
        ax[0].plot(bounds[i],[i,i],color=color,lw=2)
        ax[0].scatter(estimates[i],i,color=color,s=55,zorder=3)
    ax[0].axvline(0,color='#87968c',lw=1)
    ax[0].axvline(5,color='#a87c45',lw=1.4,ls='--',label='5% screening threshold')
    ax[0].set_yticks(range(4),['Model 3109','Model 3119','Model 3137','Pooled']); ax[0].invert_yaxis()
    ax[0].set_xlabel('Prediction-MSE reduction vs selected reference (%)')
    ax[0].set_title('A  Geometry did not pass the pooled criterion',loc='left',fontsize=12,pad=18)
    ax[0].legend(frameon=False,loc='lower right',fontsize=9)
    policy=result['policies']['0.5']
    names=['Hindsight oracle','Geometry','Current reference','Random allocation']
    vals=[policy['oracle_cost'],policy['geometry_cost'],policy['reference_cost'],policy['random_expected_cost']]
    ax[1].barh(names,vals,color=['#c5cbbf','#295d4b','#8ca294','#c5cbbf'],height=.58)
    ax[1].invert_yaxis(); ax[1].set_xlim(0,.27)
    for i,v in enumerate(vals):ax[1].text(v+.003,i,f'{v:.4f}',va='center',fontsize=10)
    ax[1].set_xlabel('Mean executed plan cost (lower is better)')
    ax[1].set_title('B  Equal budget: continue half the cases',loc='left',fontsize=12,pad=18)
    fig.suptitle('F1-A0 · Foresight geometry and the value of computation',x=.06,ha='left',fontsize=16,fontweight='bold')
    fig.text(.06,.04,'Three fixed neural world models · 300 shared evaluation scenarios · 95% paired scenario-bootstrap intervals in A\nDevelopment pilot. Geometry vs reference allocation difference: 0.00133, 95% interval [−0.00298, 0.00548].',fontsize=10,color='#526258')
    fig.subplots_adjust(left=.13,right=.97,top=.79,bottom=.23,wspace=.6)
    fig.savefig(ROOT/'F1_A0_Results.png',dpi=180)
    plt.close(fig)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
