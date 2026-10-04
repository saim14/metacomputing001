"""Independent score and decision audit using saved predictions, selections and draws."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from study_common import ROOT,SEEDS,STEPS,ARMS,DEV_ARMS,REFERENCE_ARMS,dump,sha,legacy


def verify(folder,development=False):
    arms=DEV_ARMS if development else ARMS
    axes=[0.,.025,.1] if development else STEPS
    cases=axes if development else ['natural']
    metrics=pd.read_csv(folder/'probe_metrics.csv')
    seeds=json.loads((folder/'seed_comparisons.json').read_text())
    panels=json.loads((folder/'panel_comparisons.json').read_text())
    decisions=json.loads((folder/'decisions.json').read_text())
    boot=np.load(folder/'panel_bootstrap.npz',allow_pickle=False)
    comparisons=[a for a in arms if a!='history_mlp']
    predictions={}
    row_count=0;network_runs=0;ridge_paths=0;seen=set()
    for seed in SEEDS:
        with np.load(folder/'predictions'/f'seed_{seed}.npz',allow_pickle=False) as z:
            y,p,eligible=z['target'],z['prediction'],z['eligible']
        predictions[seed]=(y,p,eligible)
        with np.load(folder/'data'/f'seed_{seed}.npz',allow_pickle=False) as z:
            x,labels=z['tokens'],z['labels']
        keys=legacy.family_keys(x)
        assert len(set(keys))==3000 and len(keys)==6000 and set(keys).isdisjoint(seen)
        seen.update(keys)
        groups=[set(keys[a:b]) for a,b in [(0,3200),(3200,4000),(4000,6000)]]
        assert all(not(a&b) for i,a in enumerate(groups) for b in groups[i+1:])
        for si,axis in enumerate(axes):
            path=folder/'models'/(f'seed_{seed}_case_{si}' if development else f'seed_{seed}_step_{axis}')
            selection=json.loads((path/'selection.json').read_text())
            chosen=min(REFERENCE_ARMS,key=lambda a:(selection[a]['validation_mse'],REFERENCE_ARMS.index(a)))
            assert selection['current_reference']['alias_of']==chosen
            np.testing.assert_array_equal(p[:,si,arms.index('current_reference')],p[:,si,arms.index(chosen)])
            if not development and axis==2:
                np.testing.assert_array_equal(p[:,si,arms.index('history_mlp')],p[:,si,arms.index('shuffled_mlp')])
            for ai,arm in enumerate(arms):
                row=metrics[(metrics.seed==seed)&(metrics.axis==axis)&(metrics.arm==arm)]
                assert len(row)==1
                row=row.iloc[0]
                np.testing.assert_allclose(row.mse,mean_squared_error(y[:,si],p[:,si,ai]),atol=1e-12)
                np.testing.assert_allclose(row.mae,mean_absolute_error(y[:,si],p[:,si,ai]),atol=1e-12)
                if np.sum((y[:,si]-y[:,si].mean())**2)>1e-12:
                    np.testing.assert_allclose(row.r2,r2_score(y[:,si],p[:,si,ai]),atol=1e-10)
                assert bool(row.eligible)==bool(eligible[si]);row_count+=1
                detail=selection[arm]
                if 'alias_of' in detail:continue
                if arm.endswith('_mlp'):
                    best=min(detail['validation_candidates'],key=lambda r:(r['validation_mse'],-r['alpha'],r['epoch']))
                    assert detail['selected_alpha']==best['alpha'] and detail['selected_epoch']==best['epoch']
                    assert len(detail['validation_candidates'])==6
                    assert detail['ensemble_parameters']==2*(32*(detail['input_dimensions']+2)+1)
                    assert detail['individual_network_training_runs']==4 and detail['minibatch_updates']==8000
                    network_runs+=4
                else:
                    best=min(detail['validation_candidates'],key=lambda r:(r['validation_mse'],-r['alpha']))
                    assert detail['selected_alpha']==best['alpha'];ridge_paths+=1
                np.testing.assert_allclose(detail['validation_mse'],best['validation_mse'],atol=1e-12)
    for row in seeds:
        y,p,eligible=predictions[row['seed']]
        idx=[axes.index(row['case'])] if development else [i for i,ok in enumerate(eligible) if ok and (row['baseline']!='shuffled_mlp' or STEPS[i]>=3)]
        assert row['axes']==[axes[i] for i in idx]
        if not idx:
            assert row['relative_mse_gain'] is None;continue
        base=p[:,:,arms.index(row['baseline'])];hist=p[:,:,arms.index('history_mlp')]
        gains=[];dr=[];ma=[]
        for i in idx:
            e0=mean_squared_error(y[:,i],base[:,i]);e1=mean_squared_error(y[:,i],hist[:,i])
            gains.append((e0-e1)/e0)
            dr.append(r2_score(y[:,i],hist[:,i])-r2_score(y[:,i],base[:,i]))
            ma.append(mean_absolute_error(y[:,i],base[:,i])-mean_absolute_error(y[:,i],hist[:,i]))
        for key,value in [('relative_mse_gain',np.mean(gains)),('delta_r2',np.mean(dr)),('mae_advantage',np.mean(ma))]:
            np.testing.assert_allclose(row[key],value,atol=1e-12)
    for row in panels:
        rows=[r for r in seeds if r['case']==row['case'] and r['baseline']==row['baseline'] and r['relative_mse_gain'] is not None]
        assert row['seeds']==[r['seed'] for r in rows]
        mean=float(np.mean([r['relative_mse_gain'] for r in rows]))
        mae=float(np.mean([r['mae_advantage'] for r in rows]))
        dr=float(np.mean([r['delta_r2'] for r in rows]))
        ci=cases.index(row['case']);bi=comparisons.index(row['baseline'])
        draws=boot[f'case_{ci}_baseline_{bi}']
        assert draws.shape==(3,2000) and np.isfinite(draws).all()
        interval=np.percentile(draws[0],[2.5,97.5])
        np.testing.assert_allclose(row['conditional_gain_ci95'],interval,atol=1e-12)
        np.testing.assert_allclose(row['conditional_delta_r2_ci95'],np.percentile(draws[1],[2.5,97.5]),atol=1e-12)
        np.testing.assert_allclose(row['mean_relative_mse_gain'],mean,atol=1e-12)
        np.testing.assert_allclose(row['mean_delta_r2'],dr,atol=1e-12)
        np.testing.assert_allclose(row['mean_mae_advantage'],mae,atol=1e-12)
        positives=sum(r['relative_mse_gain']>0 for r in rows)
        conditions={'all_six_models_eligible':len(rows)==6,'useful_point_gain':mean>=.05,
            'lower_bound_above_zero':bool(interval[0]>0),'all_six_positive':positives==6,'mae_not_worse':mae>=0}
        for k,v in conditions.items():assert row[k]==v
        assert row['positive_seeds']==positives and row['screen_passed']==all(conditions.values())
    for decision in decisions:
        required=[r for r in panels if r['case']==decision['case'] and r['baseline'] in ['current_rff_mlp','current_reference']]
        eligible=len(required)==2 and all(r['all_six_models_eligible'] for r in required)
        passed=eligible and all(r['screen_passed'] for r in required)
        excluded=eligible and all(r['conditional_gain_ci95'][1]<.05 for r in required)
        expected='ineligible' if not eligible else 'advantage_worth_independent_replication' if passed else 'useful_gain_excluded_for_this_fixed_panel_and_predictor' if excluded else 'inconclusive'
        assert decision['classification']==expected and decision['screen_passed']==passed
    assert row_count==(108 if development else 216)
    assert network_runs==(216 if development else 456) and ridge_paths==(36 if development else 72)
    if development:
        result=json.loads((folder/'development_result.json').read_text())
        assert result['go_to_main']==(not decisions[0]['screen_passed'] and decisions[-1]['screen_passed'])
    else:
        result=json.loads((folder/'study_result.json').read_text())
        assert result['decision']==decisions[0]
        for name,digest in json.loads((folder/'FIT_MANIFEST.json').read_text()).items():assert sha(folder/name)==digest,name
    return {'status':'passed','phase':'development' if development else 'main',
        'metric_rows_independently_recomputed':row_count,'seed_comparisons_verified':len(seeds),
        'panel_comparisons_verified':len(panels),'decisions_reproduced':len(decisions),
        'mlp_training_runs_accounted':network_runs,'ridge_paths_accounted':ridge_paths,
        'unique_families_verified':len(seen),'reference_choices_verified':len(SEEDS)*len(axes),
        'refitting_required':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,default=ROOT/'results')
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args();result=verify(args.folder,args.development)
    dump(args.folder/'result_verification.json',result);print(json.dumps(result,indent=2))
