"""F1-A0: one frozen, in-distribution development pilot. CPU / NumPy only."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / 'config.json').read_text())
G_NAMES = ['path_length_mean', 'path_length_sd', 'turning_mean', 'turning_sd',
           'endpoint_goal_mean', 'endpoint_goal_sd', 'clearance_min', 'clearance_mean',
           'intersection_fraction', 'endpoint_dispersion', 'efficiency_mean', 'efficiency_sd']
ARMS = ['current', 'expanded_current', 'geometry', 'raw_rollouts', 'mismatched_geometry']


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def physics(x, a):
    drift = np.stack([.25*np.sin(1.8*x[..., 1]) + .08*x[..., 0],
                      .20*np.sin(1.5*x[..., 0]) - .10*x[..., 1]], axis=-1)
    return x + .25*(a + drift)


class World:
    def __init__(self, weights):
        self.w = weights

    def __call__(self, x, a):
        shape = x.shape
        z = np.concatenate([x, a], axis=-1).reshape(-1, 4)
        z = (z-self.w['xm'])/self.w['xs']
        for i in range(3):
            z = z@self.w[f'w{i}'] + self.w[f'b{i}']
            if i < 2:
                z = np.tanh(z)
        return x + (z*self.w['ys'] + self.w['ym']).reshape(shape)


def rollout(fn, start, actions):
    x = np.broadcast_to(start[:, None, :], actions.shape[:2]+(2,)).copy()
    states = [x]
    for h in range(actions.shape[2]):
        x = fn(x, actions[:, :, h, :])
        states.append(x)
    return np.stack(states, axis=2)


def cost(paths, actions, scenarios):
    pos = paths[:, :, 1:]
    goal = scenarios[:, None, None, 2:4]
    obstacle = scenarios[:, None, None, 4:6]
    radius = scenarios[:, None, None, 6]
    d2 = np.sum((pos-goal)**2, axis=-1)
    penetration = np.maximum(radius-np.linalg.norm(pos-obstacle, axis=-1), 0)/radius
    return d2[:, :, -1]+.03*d2.mean(axis=-1)+12*(penetration**2).mean(axis=-1)+.01*(actions**2).sum(axis=-1).mean(axis=-1)


def geometry(paths, scenarios):
    delta = np.diff(paths, axis=2)
    lengths = np.linalg.norm(delta, axis=-1)
    total = lengths.sum(axis=-1)
    dot = np.sum(delta[:, :, 1:]*delta[:, :, :-1], axis=-1)
    denom = lengths[:, :, 1:]*lengths[:, :, :-1]
    turns = np.arccos(np.clip(dot/np.maximum(denom, 1e-12), -1, 1)).mean(axis=-1)
    end = paths[:, :, -1]
    distance = np.linalg.norm(end-scenarios[:, None, 2:4], axis=-1)
    obstacle = scenarios[:, None, None, 4:6]
    t = np.clip(np.sum((obstacle-paths[:, :, :-1])*delta, axis=-1)/np.maximum(lengths**2, 1e-12), 0, 1)
    nearest = paths[:, :, :-1]+t[..., None]*delta
    clearance = np.linalg.norm(nearest-obstacle, axis=-1).min(axis=-1)-scenarios[:, None, 6]
    dispersion = np.sqrt(((end-end.mean(axis=1, keepdims=True))**2).sum(axis=-1).mean(axis=1))
    efficiency = np.linalg.norm(end-paths[:, :, 0], axis=-1)/np.maximum(total, 1e-12)
    return np.stack([total.mean(1), total.std(1), turns.mean(1), turns.std(1),
                     distance.mean(1), distance.std(1), clearance.min(1), clearance.mean(1),
                     (clearance<0).mean(1), dispersion, efficiency.mean(1), efficiency.std(1)], axis=1)


def make_scenarios(n):
    rng = np.random.default_rng(42001)
    rows = []
    while len(rows) < n:
        start, goal = rng.uniform(-1.25, 1.25, (2, 2))
        if np.linalg.norm(start-goal) < .75:
            continue
        obstacle = (start+goal)/2 + rng.normal(0, .35, 2)
        radius = rng.uniform(.15, .35)
        if min(np.linalg.norm(start-obstacle), np.linalg.norm(goal-obstacle)) <= radius+.08:
            continue
        rows.append(np.r_[start, goal, obstacle, radius])
    return np.array(rows)


def train_world(seed, output):
    rng = np.random.default_rng(41001)
    x = np.c_[rng.uniform(-4, 4, (CFG['n_dynamics_train'], 2)), rng.uniform(-1, 1, (CFG['n_dynamics_train'], 2))]
    y = physics(x[:, :2], x[:, 2:])-x[:, :2]
    w = dict(xm=x.mean(0), xs=x.std(0), ym=y.mean(0), ys=y.std(0))
    nn = MLPRegressor(hidden_layer_sizes=tuple(CFG['hidden']), activation='tanh', solver='adam',
                      alpha=.0001, batch_size=256, learning_rate_init=.003,
                      max_iter=CFG['epochs'], tol=0, n_iter_no_change=100000,
                      early_stopping=False, random_state=seed)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        nn.fit((x-w['xm'])/w['xs'], (y-w['ym'])/w['ys'])
    assert nn.n_iter_ == CFG['epochs']
    for i, (a, b) in enumerate(zip(nn.coefs_, nn.intercepts_)):
        w[f'w{i}'], w[f'b{i}'] = a, b
    np.savez_compressed(output/f'world_{seed}.npz', **w)
    model = World(w)
    rng = np.random.default_rng(41002)
    xv, av = rng.uniform(-4, 4, (1024, 2)), rng.uniform(-1, 1, (1024, 2))
    pred = model(xv, av)
    sklearn_pred = xv+nn.predict((np.c_[xv, av]-w['xm'])/w['xs'])*w['ys']+w['ym']
    assert np.allclose(pred, sklearn_pred, atol=1e-12)
    s = rng.uniform(-1.25, 1.25, (256, 2))
    a = rng.uniform(-1, 1, (256, 1, CFG['horizon'], 2))
    rmse = float(np.sqrt(np.mean((rollout(model, s, a)[:, :, 1:]-rollout(physics, s, a)[:, :, 1:])**2)))
    return model, {'seed': seed, 'epochs': nn.n_iter_, 'one_step_rmse': float(np.sqrt(np.mean((pred-physics(xv,av))**2))),
                   'ten_step_rmse': rmse, 'ready': rmse < .25, 'training_seconds': time.perf_counter()-started}


def plan(world, scenarios, output, seed):
    n, p, h = len(scenarios), CFG['candidates'], CFG['horizon']
    rng = np.random.default_rng(43001)
    noise = rng.normal(size=(CFG['rounds'], n, p, h, 2))
    all_c, all_g, all_r, all_y, all_base, all_final, all_round_costs = [], [], [], [], [], [], []
    all_u2, all_u6 = [], []
    times = {'first_two_rounds_seconds': 0., 'extra_four_rounds_seconds': 0., 'geometry_seconds': 0.}
    for lo in range(0, n, 100):
        sc = scenarios[lo:lo+100]
        m = len(sc)
        mean = np.repeat(np.clip((sc[:, 2:4]-sc[:, :2])/(.25*h), -.8, .8)[:, None, :], h, axis=1)
        sd = np.full_like(mean, .65)
        incumbent = mean.copy()
        saved_costs = []
        for k in range(CFG['rounds']):
            start = time.perf_counter()
            a = np.clip(mean[:, None]+sd[:, None]*noise[k, lo:lo+m], -1, 1)
            a[:, 0], a[:, 1] = incumbent, mean
            paths = rollout(world, sc[:, :2], a)
            values = cost(paths, a, sc)
            order = np.argsort(values, axis=1, kind='stable')
            incumbent = a[np.arange(m), order[:, 0]].copy()
            best = values[np.arange(m), order[:, 0]]
            saved_costs.append(best)
            elite = a[np.arange(m)[:, None], order[:, :CFG['elites']]]
            mean, sd = elite.mean(axis=1), np.maximum(elite.std(axis=1), .08)
            elapsed = time.perf_counter()-start
            times['first_two_rounds_seconds' if k<2 else 'extra_four_rounds_seconds'] += elapsed
            if k+1 == CFG['decision_round']:
                c = np.c_[sc, mean.reshape(m,-1), sd.reshape(m,-1), incumbent.reshape(m,-1),
                          np.sort(values,axis=1), best]
                t = time.perf_counter()
                g = geometry(paths, sc)
                times['geometry_seconds'] += time.perf_counter()-t
                all_c.append(c); all_g.append(g); all_r.append(paths.reshape(m,-1).copy())
                u2 = incumbent.copy()
        costs = np.stack(saved_costs, axis=1)
        assert np.max(np.diff(costs, axis=1)) < 1e-10
        base = cost(rollout(physics, sc[:, :2], u2[:, None]), u2[:, None], sc)[:, 0]
        final = cost(rollout(physics, sc[:, :2], incumbent[:, None]), incumbent[:, None], sc)[:, 0]
        all_y.append(base-final); all_base.append(base); all_final.append(final); all_round_costs.append(costs)
        all_u2.append(u2); all_u6.append(incumbent.copy())
    data = dict(current=np.concatenate(all_c), geometry=np.concatenate(all_g), raw=np.concatenate(all_r),
                target=np.concatenate(all_y), base=np.concatenate(all_base), final=np.concatenate(all_final),
                model_costs=np.concatenate(all_round_costs), u2=np.concatenate(all_u2), u6=np.concatenate(all_u6))
    np.savez_compressed(output/f'planning_{seed}.npz', **data)
    return data, times


def scale(x, idx):
    mean, sd = x[idx].mean(0), x[idx].std(0)
    sd[sd<1e-9] = 1
    return (x-mean)/sd, mean, sd


def arms(data, fit, splits, mi):
    c, g = data['current'], data['geometry']
    z, _, _ = scale(c, fit)
    rng = np.random.default_rng(44001)
    expanded = np.sqrt(2/12)*np.cos(z@rng.normal(size=(c.shape[1],12))/np.sqrt(c.shape[1])+rng.uniform(0,2*np.pi,12))
    wrong = np.empty_like(g)
    rng = np.random.default_rng(44002+mi)
    for idx in splits:
        wrong[idx] = g[rng.permutation(idx)]
    return dict(current=c, expanded_current=np.c_[c,expanded], geometry=np.c_[c,g],
                raw_rollouts=np.c_[c,data['raw']], mismatched_geometry=np.c_[c,wrong])


def fit_arm(x, y, fit, tune, output, seed, arm):
    z, xm, xs = scale(x, fit)
    ym = y[fit].mean()
    candidate_rows, best = [], None
    for kind in ['linear']+CFG['bandwidths']:
        if kind == 'linear':
            basis, w, phase = z, np.empty((0,0)), np.empty(0)
        else:
            rng = np.random.default_rng(45001)
            w = rng.normal(size=(x.shape[1], CFG['rff_width']))/(np.sqrt(x.shape[1])*kind)
            phase = rng.uniform(0,2*np.pi,CFG['rff_width'])
            basis = np.sqrt(2/CFG['rff_width'])*np.cos(z@w+phase)
        b, bm, bs = scale(basis, fit)
        gram = b[fit].T@b[fit]
        vals, vecs = eigh(gram, check_finite=False)
        vals = np.maximum(vals,0)
        rhs = vecs.T@(b[fit].T@(y[fit]-ym))
        for alpha in CFG['alphas']:
            beta = vecs@(rhs/(vals+len(fit)*alpha))
            pred = b@beta+ym
            mse = float(np.mean((pred[tune]-y[tune])**2))
            row = {'kind':str(kind),'alpha':alpha,'tuning_mse':mse,'coefficients':len(beta)+1}
            candidate_rows.append(row)
            if best is None or mse<best['tuning_mse']:
                best = dict(row, predictions=pred.copy(), state=dict(xm=xm,xs=xs,bm=bm,bs=bs,w=w,phase=phase,beta=beta,ym=np.array(ym)))
    np.savez_compressed(output/f'observer_{seed}_{arm}.npz', **best['state'])
    state=best.pop('state')
    assert len(candidate_rows)==9
    return best, candidate_rows


def ci(values, boot):
    v = np.asarray(values).mean(axis=0)
    means = v[boot].mean(axis=1)
    return {'mean':float(v.mean()), 'ci95':[float(x) for x in np.quantile(means,[.025,.975])]}


def allocated(pred, y, base, fraction):
    result, masks = [], []
    for p, v, b in zip(pred,y,base):
        mask=np.zeros(len(v),dtype=bool)
        mask[np.argsort(-p,kind='stable')[:int(len(v)*fraction)]]=True
        result.append(b-v*mask); masks.append(mask)
    return np.array(result), np.array(masks)


def fixtures():
    rng=np.random.default_rng(999001)
    sc=make_scenarios(5)
    a=rng.uniform(-1,1,(5,4,10,2))
    paths=rollout(physics,sc[:,:2],a)
    g=geometry(paths,sc)
    angle=.731
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    offset=np.array([2.,-1.])
    sr=sc.copy()
    for i in [0,2,4]:sr[:,i:i+2]=sc[:,i:i+2]@rotation.T+offset
    err=float(np.max(np.abs(g-geometry(paths@rotation.T+offset,sr))))
    assert err<1e-8
    assert g.shape==(5,12)
    assert np.isfinite(g).all()
    return {'geometry_rigid_transform_max_error':err,'geometry_dimensions':12}


def run():
    output=ROOT/'run'
    if output.exists():
        raise RuntimeError('A run directory already exists. Preserve it; do not silently rerun this pilot.')
    output.mkdir()
    frozen={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['F1_A0_Protocol.md','config.json','run_f1.py']}
    manifest={'study':'F1-A0','protocol_version':'1.0','started_utc':datetime.now(timezone.utc).isoformat(),
              'source_sha256':frozen,'versions':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
              'fixtures':fixtures()}
    dump(output/'manifest.json',manifest)
    sc=make_scenarios(CFG['n_scenarios'])
    order=np.random.default_rng(42002).permutation(len(sc))
    fit,tune,test=np.split(order,[CFG['n_fit'],CFG['n_fit']+CFG['n_tune']])
    assert not(set(fit)&set(tune) or set(fit)&set(test) or set(tune)&set(test))
    np.savez_compressed(output/'scenarios.npz',scenarios=sc,fit=fit,tune=tune,evaluation=test)
    worlds=[]; readiness=[]
    for seed in CFG['model_seeds']:
        model,record=train_world(seed,output)
        worlds.append(model); readiness.append(record)
        print(json.dumps({'stage':'world_model',**record}),flush=True)
    dump(output/'world_readiness.json',readiness)
    if not all(r['ready'] for r in readiness):
        dump(output/'result.json',{'status':'world_model_readiness_failed','readiness':readiness})
        return
    datasets=[]; timings=[]; target_checks=[]
    for seed,world in zip(CFG['model_seeds'],worlds):
        data,timing=plan(world,sc,output,seed)
        datasets.append(data); timings.append(timing)
        checks={name:{'sd':float(data['target'][idx].std()),'beneficial_count':int((data['target'][idx]>.01).sum())} for name,idx in [('fit',fit),('tuning',tune)]}
        checks['ready']=all(c['sd']>=.01 and c['beneficial_count']>=25 for c in checks.values())
        target_checks.append(checks)
        print(json.dumps({'stage':'planning','seed':seed,'checks':checks,'timing':timing}),flush=True)
    dump(output/'target_readiness.json',target_checks)
    if not all(r['ready'] for r in target_checks):
        dump(output/'result.json',{'status':'uninformative_target','readiness':readiness,'target_checks':target_checks})
        return
    predictions={a:[] for a in ARMS}; references=[]; selections=[]; grid=[]; inference=[]
    for mi,(seed,data) in enumerate(zip(CFG['model_seeds'],datasets)):
        matrices=arms(data,fit,[fit,tune,test],mi)
        chosen={}
        for arm in ARMS:
            t=time.perf_counter()
            best,rows=fit_arm(matrices[arm],data['target'],fit,tune,output,seed,arm)
            predictions[arm].append(best.pop('predictions')[test])
            chosen[arm]=dict(best,input_dimensions=matrices[arm].shape[1],fit_seconds=time.perf_counter()-t)
            grid.extend([dict(seed=seed,arm=arm,**row) for row in rows])
        ref=min(['current','expanded_current','raw_rollouts'],key=lambda a:chosen[a]['tuning_mse'])
        references.append(ref); selections.append(chosen)
        print(json.dumps({'stage':'observers','seed':seed,'strong_reference':ref,'selection':chosen}),flush=True)
    predictions={k:np.array(v) for k,v in predictions.items()}
    y=np.array([d['target'][test] for d in datasets])
    base=np.array([d['base'][test] for d in datasets])
    final=np.array([d['final'][test] for d in datasets])
    ref=np.array([predictions[name][mi] for mi,name in enumerate(references)])
    geo=predictions['geometry']
    err_ref,err_geo=(ref-y)**2,(geo-y)**2
    boot=np.random.default_rng(46001).integers(0,len(test),(CFG['bootstrap'],len(test)))
    primary=ci(err_ref-err_geo,boot)
    primary['relative_mse_reduction']=float((err_ref.mean()-err_geo.mean())/err_ref.mean())
    mismatch=ci((predictions['mismatched_geometry']-y)**2-err_geo,boot)
    policies={}
    for q in [.25,.5,.75]:
        lg,mg=allocated(geo,y,base,q)
        lr,mr=allocated(ref,y,base,q)
        oracle,_=allocated(y,y,base,q)
        actual_quota=int(len(test)*q)/len(test)
        random_expected=base-actual_quota*y
        policies[str(q)]={'geometry_cost':float(lg.mean()),'reference_cost':float(lr.mean()),
                          'random_expected_cost':float(random_expected.mean()),'oracle_cost':float(oracle.mean()),
                          'benefit_vs_reference':ci(lr-lg,boot),'benefit_vs_random':ci(random_expected-lg,boot),
                          'extra_rounds_per_case':4*actual_quota}
    per_model=[]
    for mi,seed in enumerate(CFG['model_seeds']):
        constant=float(datasets[mi]['target'][fit].mean())
        per_model.append({'seed':seed,'strong_reference':references[mi],
                          'arm_mse':{a:float(np.mean((predictions[a][mi]-y[mi])**2)) for a in ARMS},
                          'constant_mse':float(np.mean((constant-y[mi])**2)),
                          'reference_mse':float(err_ref[mi].mean()),'geometry_mse':float(err_geo[mi].mean()),
                          'relative_mse_reduction':float(1-err_geo[mi].mean()/err_ref[mi].mean()),
                          'beneficial_fraction':float((y[mi]>.01).mean()),'harmful_fraction':float((y[mi]<-.01).mean())})
    criteria={'relative_mse_at_least_5pct':primary['relative_mse_reduction']>=.05,
              'positive_primary_lower_interval':primary['ci95'][0]>0,
              'each_model_positive':all(m['relative_mse_reduction']>0 for m in per_model),
              'positive_mismatch_lower_interval':mismatch['ci95'][0]>0,
              'positive_allocation_lower_interval':policies['0.5']['benefit_vs_reference']['ci95'][0]>0}
    result={'study':'F1-A0','protocol_version':'1.0','status':'encouraging_development_pilot' if all(criteria.values()) else 'combined_advancement_criterion_not_met',
            'world_readiness':readiness,'target_readiness':target_checks,'strong_references':references,
            'primary_mse_difference':primary,'mismatched_mse_difference':mismatch,'criteria':criteria,
            'per_model':per_model,'pooled_arm_mse':{a:float(np.mean((p-y)**2)) for a,p in predictions.items()},
            'reference_mse':float(err_ref.mean()),'geometry_mse':float(err_geo.mean()),'policies':policies,
            'always_stop_cost':float(base.mean()),'always_continue_cost':float(final.mean()),
            'selected_observers':selections,'candidate_fits':len(grid),'timings':timings,'geometry_feature_names':G_NAMES,
            'finished_utc':datetime.now(timezone.utc).isoformat()}
    assert len(grid)==135
    np.savez_compressed(output/'evaluation.npz',scenario_ids=test,target=y,base=base,final=final,reference=ref,**predictions)
    dump(output/'candidate_grid.json',grid)
    dump(output/'result.json',result)
    assert frozen=={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in frozen}
    print(json.dumps({'stage':'finished','status':result['status'],'primary':primary,'policy_50':policies['0.5'],'criteria':criteria}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixtures-only',action='store_true')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.fixtures_only:
            print(json.dumps(fixtures()))
        else:
            run()
