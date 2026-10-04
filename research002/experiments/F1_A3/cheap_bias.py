"""F1-A3 methods prototype: compact residual correction and fixed candidate selection."""
import numpy as np

def quadratic(x):
    x=np.asarray(x,dtype=float)
    if x.shape[-1]!=4:raise ValueError('Expected two state and two action coordinates')
    terms=[np.ones(x.shape[:-1])]+[x[...,i] for i in range(4)]
    terms += [x[...,i]*x[...,j] for i in range(4) for j in range(i,4)]
    return np.stack(terms,axis=-1)

class CompactResidual:
    """A fixed quadratic ridge map, fit to independent transition residuals only."""
    def __init__(self,x,residual,ridge=1e-3):
        self.scale=np.array([4.,4.,1.,1.]);design=quadratic(np.asarray(x)/self.scale)
        penalty=np.eye(design.shape[1])*ridge;penalty[0,0]=0
        self.coef=np.linalg.solve(design.T@design+penalty,design.T@np.asarray(residual))
    def correct(self,x,a):
        return quadratic(np.concatenate([x,a],axis=-1)/self.scale)@self.coef

def subset_indices(model_cost,k=8):
    """Evenly spaced ranks of current model costs; no true costs or future plans."""
    costs=np.asarray(model_cost)
    if costs.ndim!=2 or not 2<=k<=costs.shape[1]:raise ValueError('Invalid candidate costs or subset size')
    ranks=np.rint(np.linspace(0,costs.shape[1]-1,k)).astype(int)
    return np.argsort(costs,axis=1,kind='stable')[:,ranks]

def fixtures():
    rng=np.random.default_rng(73001);x=rng.uniform(-1,1,(256,4));w=rng.normal(0,.01,(4,2))
    b=np.array([.03,-.02]);map_=CompactResidual(x,x@w+b,ridge=1e-8)
    q=rng.uniform(-1,1,(32,4));error=np.max(np.abs(map_.correct(q[:,:2],q[:,2:])-(q@w+b)))
    assert error<1e-6
    costs=rng.normal(size=(5,32));ids=subset_indices(costs)
    assert ids.shape==(5,8) and np.array_equal(ids[:,0],np.argmin(costs,axis=1))
    assert all(len(set(row))==8 for row in ids)
    return {'affine_max_error':float(error),'subset_contains_incumbent':True,'unique_candidates':True}

if __name__=='__main__':
    import json
    print(json.dumps(fixtures(),indent=2))
