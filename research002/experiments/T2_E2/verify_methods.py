"""Bounded synthetic checks of inference, serialization, controls, and aggregation."""
import json
import tempfile
from pathlib import Path
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from threadpoolctl import threadpool_limits
from observer import fit_mlp,predict,network_predict,standardizer
from study_common import ROOT,SEEDS,STEPS,ARMS,matrices,transformations,dump,verify_inputs
from aggregate_results import aggregate


def verify():
    verify_inputs()
    rng=np.random.default_rng(509871)
    x=rng.normal(size=(640,8));y=np.maximum(x[:,0],0)+.2*rng.normal(size=640)
    model,detail=fit_mlp(x[:512],y[:512],x[512:],y[512:],seed=1,step=2)
    pp=predict(model,x[512:])
    np.testing.assert_allclose(mean_squared_error(y[512:],pp),detail['validation_mse'],atol=1e-12)
    assert detail['ensemble_parameters']==642 and detail['minibatch_updates']==1280
    assert len(detail['validation_candidates'])==6
    selected=min(detail['validation_candidates'],key=lambda r:(r['validation_mse'],-r['alpha'],r['epoch']))
    assert detail['selected_alpha']==selected['alpha'] and detail['selected_epoch']==selected['epoch']
    direct=MLPRegressor(hidden_layer_sizes=(32,),max_iter=1,random_state=2)
    direct.partial_fit(x.astype(np.float32),y.astype(np.float32))
    weights=[direct.coefs_[0],direct.intercepts_[0],direct.coefs_[1],direct.intercepts_[1]]
    np.testing.assert_allclose(network_predict(x.astype(np.float32),weights),direct.predict(x.astype(np.float32)),atol=1e-7)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'probe.npz';np.savez_compressed(p,**model)
        with np.load(p,allow_pickle=False) as z:loaded={k:z[k] for k in z.files}
        np.testing.assert_array_equal(pp,predict(loaded,x[512:]))
    mean,sd=standardizer(x[:512]);np.testing.assert_array_equal(model['x_mean'],mean)
    np.testing.assert_array_equal(model['x_sd'],sd)
    current=rng.normal(size=(6000,388));past=rng.normal(size=(6000,108))
    tf=transformations(current,past,1,4,'main')
    views=matrices(current,past,tf,1,4,'main')
    shuffled=views['shuffled'][:,388:].reshape(3000,2,3,36)
    original=past.reshape(3000,2,3,36)
    for family in [0,1599,1600,1999,2000,2999]:
        permutation=[int(np.flatnonzero(np.all(original[family,0]==row,axis=1))[0]) for row in shuffled[family,0]]
        np.testing.assert_array_equal(shuffled[family,1],original[family,1,permutation])
    split_views=matrices(current[:4000],past[:4000],tf,1,4,'main')
    test_views=matrices(current[4000:],past[4000:],tf,1,4,'main',partition_start=2)
    for arm in views:
        np.testing.assert_allclose(views[arm][:4000],split_views[arm],atol=1e-12)
        np.testing.assert_allclose(views[arm][4000:],test_views[arm],atol=1e-12)
    with tempfile.TemporaryDirectory() as td:
        folder=Path(td);(folder/'predictions').mkdir()
        for seed in SEEDS:
            yy=rng.normal(size=(2000,4));ee=rng.normal(size=(2000,4))
            predictions=np.repeat((yy+ee)[:,:,None],len(ARMS),axis=2)
            predictions[:,:,ARMS.index('history_mlp')]=yy+.8*ee
            np.savez_compressed(folder/'predictions'/f'seed_{seed}.npz',target=yy,prediction=predictions,
                eligible=np.array([True,False,True,True]))
        panels,decisions=aggregate(folder)
        for panel in panels:
            np.testing.assert_allclose(panel['mean_relative_mse_gain'],.36,atol=1e-12)
            np.testing.assert_allclose(panel['conditional_gain_ci95'],[.36,.36],atol=1e-12)
        assert decisions[0]['screen_passed']
        for row in json.loads((folder/'seed_comparisons.json').read_text()):
            assert row['axes']==([4,5] if row['baseline']=='shuffled_mlp' else [2,4,5])
    return {'status':'passed','mlp_serialization':'exact','manual_forward_vs_sklearn':'passed',
        'selected_ensemble_validation_mse':'reproduced','training_only_standardizer':'verified',
        'parameter_and_update_counts':'verified','temporal_permutation_pair_consistency':'passed',
        'separate_test_control_generation':'matches_combined_generation',
        'known_36_percent_gain_and_bootstrap_interval':'reproduced_for_all_comparisons',
        'eligibility_and_chronology_filtering':'verified','research_datasets_used':0,
        'synthetic_mlp_training_runs':4,'direct_sklearn_fixture_runs':1}


if __name__=='__main__':
    with threadpool_limits(limits=1):result=verify()
    dump(ROOT/'method_verification.json',result)
    print(json.dumps(result,indent=2))
