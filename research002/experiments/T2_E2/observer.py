"""Fixed small MLP ensembles and ridge references; fitting accepts train/validation only."""
from __future__ import annotations
import time
import numpy as np
from sklearn.neural_network import MLPRegressor

ALPHAS = [.1, 10.]
EPOCHS = [20, 40, 80]
RIDGE_ALPHAS = [.01, .1, 1., 10., 100.]


def standardizer(x):
    x = np.asarray(x, np.float64)
    return x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)


def network_predict(z, weights):
    w1, b1, w2, b2 = weights
    return (np.maximum(z @ w1 + b1, 0) @ w2 + b2).ravel()


def predict(bundle, x):
    z = ((np.asarray(x, np.float64)-bundle['x_mean'])/bundle['x_sd']).astype(np.float32)
    if bundle['kind'].item() == 'mlp':
        predictions = [network_predict(z, [bundle[f'n{i}_{key}'] for key in ['w1','b1','w2','b2']]) for i in range(2)]
        return np.mean(predictions, axis=0).astype(np.float64)*bundle['y_sd']+bundle['y_mean']
    z = (np.asarray(x, np.float64)-bundle['x_mean'])/bundle['x_sd']
    return z @ bundle['coef']+bundle['y_mean']


def fit_mlp(x_train, y_train, x_val, y_val, *, seed, step, phase_offset=0):
    start = time.time()
    mean, sd = standardizer(x_train)
    z = ((x_train-mean)/sd).astype(np.float32)
    zv = ((x_val-mean)/sd).astype(np.float32)
    ym, ys = float(np.mean(y_train)), max(float(np.std(y_train)), 1e-6)
    y = ((y_train-ym)/ys).astype(np.float32)
    rng = np.random.default_rng(440000000+phase_offset+seed*100+step)
    orders = [rng.permutation(len(y)) for _ in range(80)]
    candidates, losses = {}, []
    for alpha in ALPHAS:
        snapshots = {}
        for initialization in range(2):
            model = MLPRegressor(hidden_layer_sizes=(32,), activation='relu', solver='adam',
                alpha=alpha, batch_size=128, learning_rate_init=.001, max_iter=1,
                shuffle=False, random_state=430000000+phase_offset+seed*1000+step*10+initialization,
                early_stopping=False, beta_1=.9, beta_2=.999, epsilon=1e-8)
            for epoch, order in enumerate(orders, 1):
                model.partial_fit(z[order], y[order])
                if epoch in EPOCHS:
                    weights = [model.coefs_[0].copy(), model.intercepts_[0].copy(),
                               model.coefs_[1].copy(), model.intercepts_[1].copy()]
                    snapshots[(initialization, epoch)] = weights
            assert model.t_ == 80*len(y)
        for epoch in EPOCHS:
            weights = [snapshots[(i, epoch)] for i in range(2)]
            pp = np.mean([network_predict(zv, w) for w in weights], axis=0).astype(np.float64)*ys+ym
            loss = float(np.mean((pp-y_val)**2))
            losses.append({'alpha': alpha, 'epoch': epoch, 'validation_mse': loss})
            candidates[(alpha, epoch)] = weights
    best = min(losses, key=lambda r: (r['validation_mse'], -r['alpha'], r['epoch']))
    bundle = {'kind': np.array('mlp'), 'x_mean': mean, 'x_sd': sd,
              'y_mean': np.array(ym), 'y_sd': np.array(ys)}
    for i, weights in enumerate(candidates[(best['alpha'], best['epoch'])]):
        bundle.update({f'n{i}_{key}': arr for key, arr in zip(['w1','b1','w2','b2'], weights)})
    assert all(np.isfinite(a).all() for k,a in bundle.items() if k != 'kind')
    count = 2*(32*(x_train.shape[1]+2)+1)
    assert sum(a.size for k,a in bundle.items() if k.startswith('n')) == count
    details = {'selected_alpha':best['alpha'], 'selected_epoch':best['epoch'],
        'validation_mse':best['validation_mse'], 'validation_candidates':losses,
        'input_dimensions':x_train.shape[1], 'ensemble_parameters':count,
        'affine_macs_per_example':2*32*(x_train.shape[1]+1),
        'individual_network_training_runs':4, 'epochs_trained':320,
        'minibatch_updates':320*int(np.ceil(len(y)/128)), 'fit_seconds':time.time()-start}
    return bundle, details


def fit_ridge(x_train, y_train, x_val, y_val):
    start = time.time()
    mean, sd = standardizer(x_train)
    z, zv = (x_train-mean)/sd, (x_val-mean)/sd
    ym = float(np.mean(y_train))
    vals, vectors = np.linalg.eigh(z.T @ z)
    vals = np.maximum(vals, 0)
    rhs = vectors.T @ (z.T @ (y_train-ym))
    coefs = vectors @ (rhs[:, None]/(vals[:, None]+np.array(RIDGE_ALPHAS)[None, :]))
    vp = zv @ coefs+ym
    losses = ((vp-y_val[:, None])**2).mean(axis=0)
    j = min(range(len(RIDGE_ALPHAS)), key=lambda i:(losses[i], -RIDGE_ALPHAS[i]))
    bundle = {'kind':np.array('ridge'),'x_mean':mean,'x_sd':sd,'y_mean':np.array(ym),'coef':coefs[:,j]}
    return bundle, {'selected_alpha':RIDGE_ALPHAS[j], 'validation_mse':float(losses[j]),
        'validation_candidates':[{'alpha':a,'validation_mse':float(v)} for a,v in zip(RIDGE_ALPHAS,losses)],
        'input_dimensions':x_train.shape[1], 'ensemble_parameters':x_train.shape[1]+1,
        'affine_macs_per_example':x_train.shape[1], 'fit_seconds':time.time()-start}
