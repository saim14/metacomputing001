"""Paired-family uncertainty, fixed-seed aggregation, and frozen decision rules."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from study_common import SEEDS,STEPS,ARMS,DEV_ARMS,dump


def score(y,p):
    sst=np.sum((y-y.mean())**2)
    return {'mse':float(np.mean((y-p)**2)), 'mae':float(np.abs(y-p).mean()),
            'r2':float(1-np.sum((y-p)**2)/sst) if sst>1e-12 else None}


def aggregate(folder, development=False):
    arms=DEV_ARMS if development else ARMS
    axes=[0.,.025,.1] if development else STEPS
    comparisons=[a for a in arms if a!='history_mlp']
    cases=axes if development else ['natural']
    metrics=[];seed_rows=[];draws_by_case={case:{} for case in cases}
    rng=np.random.default_rng(460000000+(0 if development else 1000000))
    for seed in SEEDS:
        with np.load(folder/'predictions'/f'seed_{seed}.npz',allow_pickle=False) as z:
            y,p,eligible=z['target'],z['prediction'],z['eligible']
        assert p.shape==(2000,len(axes),len(arms)) and y.shape==(2000,len(axes))
        assert np.isfinite(y).all() and np.isfinite(p).all()
        for i,axis in enumerate(axes):
            for ai,arm in enumerate(arms):
                metrics.append({'seed':seed,'axis':axis,'arm':arm,'eligible':bool(eligible[i]),**score(y[:,i],p[:,i,ai])})
        counts=rng.multinomial(1000,np.full(1000,.001),size=2000).astype(np.float64)
        hist=p[:,:,arms.index('history_mlp')]
        herror=(y-hist)**2
        ysum=y.reshape(1000,2,len(axes)).sum(axis=1)
        y2sum=(y*y).reshape(1000,2,len(axes)).sum(axis=1)
        sst=((y-y.mean(axis=0))**2).sum(axis=0)
        bs_sst=counts@y2sum-(counts@ysum)**2/2000
        for case in cases:
            selected=np.array([axes.index(case)]) if development else np.flatnonzero(eligible)
            for baseline in comparisons:
                indices=selected if development or baseline!='shuffled_mlp' else selected[np.array(STEPS)[selected]>=3]
                if not len(indices):
                    seed_rows.append({'case':case,'seed':seed,'baseline':baseline,'axes':[],
                        'relative_mse_gain':None,'delta_r2':None,'mae_advantage':None});continue
                pp=p[:,:,arms.index(baseline)]
                base_error=(y-pp)**2
                base_family=base_error.reshape(1000,2,len(axes)).sum(axis=1)
                diff_family=(base_error-herror).reshape(1000,2,len(axes)).sum(axis=1)
                abs_family=(np.abs(y-pp)-np.abs(y-hist)).reshape(1000,2,len(axes)).sum(axis=1)
                denom=base_family.sum(axis=0)[indices]
                assert (denom>1e-12).all() and (sst[indices]>1e-12).all()
                bd=(counts@base_family)[:,indices]
                num=(counts@diff_family)[:,indices]
                assert (bd>1e-12).all() and (bs_sst[:,indices]>1e-12).all()
                br=(num/bd).mean(axis=1)
                bdr=(num/bs_sst[:,indices]).mean(axis=1)
                bma=(counts@abs_family)[:,indices].mean(axis=1)/2000
                point=float((diff_family.sum(axis=0)[indices]/denom).mean())
                dr=float((diff_family.sum(axis=0)[indices]/sst[indices]).mean())
                mae=float(abs_family[:,indices].sum(axis=0).mean()/2000)
                row={'case':case,'seed':seed,'baseline':baseline,'axes':[axes[i] for i in indices],
                    'relative_mse_gain':point,'delta_r2':dr,'mae_advantage':mae,
                    'conditional_gain_ci95':np.quantile(br,[.025,.975]).tolist()}
                seed_rows.append(row)
                draws_by_case[case][(seed,baseline)]=(br,bdr,bma)
    panels=[];saved_boot={}
    for ci,case in enumerate(cases):
        for bi,baseline in enumerate(comparisons):
            rows=[r for r in seed_rows if r['case']==case and r['baseline']==baseline and r['relative_mse_gain'] is not None]
            if not rows:continue
            boot=np.mean([draws_by_case[case][(r['seed'],baseline)] for r in rows],axis=0)
            saved_boot[f'case_{ci}_baseline_{bi}']=boot
            lo,hi=np.quantile(boot[0],[.025,.975])
            mean=float(np.mean([r['relative_mse_gain'] for r in rows]))
            mae=float(np.mean([r['mae_advantage'] for r in rows]))
            positives=sum(r['relative_mse_gain']>0 for r in rows)
            checks={'all_six_models_eligible':len(rows)==6,'useful_point_gain':mean>=.05,
                'lower_bound_above_zero':bool(lo>0),'all_six_positive':positives==6,'mae_not_worse':mae>=0}
            panels.append({'case':case,'baseline':baseline,'seeds':[r['seed'] for r in rows],
                'mean_relative_mse_gain':mean,'conditional_gain_ci95':[float(lo),float(hi)],
                'mean_delta_r2':float(np.mean([r['delta_r2'] for r in rows])),
                'conditional_delta_r2_ci95':np.quantile(boot[1],[.025,.975]).tolist(),
                'mean_mae_advantage':mae,'positive_seeds':positives,**checks,'screen_passed':all(checks.values())})
    np.savez_compressed(folder/'panel_bootstrap.npz',**saved_boot)
    pd.DataFrame(metrics).to_csv(folder/'probe_metrics.csv',index=False)
    dump(folder/'seed_comparisons.json',seed_rows);dump(folder/'panel_comparisons.json',panels)
    decisions=[]
    for case in cases:
        required=[r for r in panels if r['case']==case and r['baseline'] in ['current_rff_mlp','current_reference']]
        eligible=len(required)==2 and all(r['all_six_models_eligible'] for r in required)
        passed=eligible and all(r['screen_passed'] for r in required)
        excluded=eligible and all(r['conditional_gain_ci95'][1]<.05 for r in required)
        classification=('ineligible' if not eligible else 'advantage_worth_independent_replication' if passed else
                        'useful_gain_excluded_for_this_fixed_panel_and_predictor' if excluded else 'inconclusive')
        decisions.append({'case':case,'panel_eligible':eligible,'screen_passed':passed,'classification':classification})
    dump(folder/'decisions.json',decisions)
    return panels,decisions
