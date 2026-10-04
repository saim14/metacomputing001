"""Data separation, unchanged frozen model inference, and predictor observations."""
import hashlib
import json
from pathlib import Path
import numpy as np
import legacy_t2e1 as legacy
from observer import standardizer

ROOT = Path(__file__).resolve().parent
SEEDS = [503,607,709,811,907,1009]
STEPS = [2,3,4,5]
TRAIN, VAL, TEST = slice(0,3200),slice(3200,4000),slice(4000,6000)
PARTS = [TRAIN,VAL,TEST]
ARMS = ['current_mlp','current_rff_mlp','history_mlp','shuffled_mlp','mismatched_mlp',
        'current_ridge','current_rff_ridge','history_ridge','current_reference']
DEV_ARMS = ['current_mlp','current_rff_mlp','history_mlp','current_ridge','current_rff_ridge','current_reference']
REFERENCE_ARMS = ['current_mlp','current_rff_mlp','current_ridge','current_rff_ridge']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, data): Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def verify_inputs():
    for name,digest in json.loads((ROOT/'FROZEN_INPUT_SHA256.json').read_text()).items():
        assert sha(ROOT/name) == digest, name


def old_exclusions(study, seed):
    forbidden = set()
    streams = {p['data_stream_id']:{**study['data'],**p.get('data_overrides',{})} for p in study['training']['curriculum']}
    for stream,cfg in streams.items():
        for n,offset in [(cfg['train_samples'],11),(cfg['validation_samples'],23)]:
            x,_ = legacy.generate_task(n,data_config=cfg,rng=np.random.default_rng(seed*10000+stream*100+offset))
            forbidden.update(legacy.family_keys(x))
    for co,ao in ([(37,43)] if seed<811 else [(37,43),(83,89)]):
        forbidden.update(legacy.family_keys(legacy.generate_clean_evaluation(study,seed,co)['tokens']))
        suite=legacy.generate_two_hop_counterfactual_suite(2000,key_count=8,pair_count=5,rng=np.random.default_rng(seed*100+ao))
        for name,x in suite.items():
            if name.endswith('tokens'):forbidden.update(legacy.family_keys(x))
    with np.load(ROOT/'reference/exposed_T2_E1_tokens.npz',allow_pickle=False) as z:
        for key in z.files:forbidden.update(legacy.family_keys(z[key]))
    return forbidden


def new_data(study, seed, phase, prior_study_families):
    forbidden=old_exclusions(study,seed) | prior_study_families
    base=400000000 if phase=='development' else 410000000
    rng=np.random.default_rng(base+seed)
    seen=set();chosen=[];labels=[];rejected=0
    for batch in range(5):
        suite=legacy.generate_two_hop_counterfactual_suite(4000,key_count=8,pair_count=5,rng=rng)
        for i,key in enumerate(legacy.family_keys(suite['clean_tokens'])):
            if key in forbidden or key in seen:
                rejected+=1;continue
            seen.add(key)
            chosen.append(np.stack([suite['clean_tokens'][i],suite['counterfactual_tokens'][i]]))
            labels.append([suite['clean_labels'][i],suite['counterfactual_labels'][i]])
            if len(chosen)==3000:break
        if len(chosen)==3000:break
    assert len(chosen)==3000,'Fixed candidate budget exhausted'
    order=np.random.default_rng(base+1000000+seed).permutation(3000)
    x=np.asarray(chosen)[order].reshape(6000,12)
    y=np.asarray(labels,dtype=np.int64)[order].reshape(6000)
    keys=legacy.family_keys(x)
    groups=[set(keys[p][::2]) for p in PARTS]
    assert all(not(a&b) for i,a in enumerate(groups) for b in groups[i+1:])
    assert set(keys).isdisjoint(forbidden) and len(set(keys))==3000
    assert all(keys[i]==keys[i+1] for i in range(0,6000,2))
    assert all(y[p].mean()==.5 for p in PARTS)
    metadata={'seed':seed,'phase':phase,'unique_families':3000,'excluded_families':len(forbidden),
        'candidate_batches':batch+1,'rejected_candidates':rejected,'cross_split_overlap':0,
        'prior_exposure_overlap':0,'earlier_study_family_overlap':0,'pair_members_share_split':True,
        'tokens_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'labels_sha256':hashlib.sha256(y.tobytes()).hexdigest()}
    return x,y,metadata,set(keys)


def raw_views(full,cls,summaries,step):
    current=np.concatenate([full[:,step-1],summaries[:,step]],axis=1).astype(np.float64)
    history=cls[:,1:step].reshape(len(cls),-1).astype(np.float64)
    assert current.shape[1]==388 and history.shape[1]==36*(step-1)
    return current,history


def transformations(current,history,seed,step,phase):
    """Called on 4,000 main development observations or 6,000 calibration observations."""
    feature_offset=0 if phase=='development' else 1000000
    mean,sd=standardizer(current[TRAIN])
    rng=np.random.default_rng(420000000+feature_offset+seed*100+step)
    h=history.shape[1]
    transform={'mean':mean,'sd':sd,'w':rng.normal(size=(388,h))/np.sqrt(388),
               'bias':rng.uniform(0,2*np.pi,size=h)}
    return transform


def matrices(current,history,transform,seed,step,phase,partition_start=0):
    n=len(current);slots=step-1
    rff=np.sqrt(2/history.shape[1])*np.cos(((current-transform['mean'])/transform['sd'])@transform['w']+transform['bias'])
    raw=history.reshape(n,slots,36)
    shuffled=raw.copy();mismatched=raw.copy()
    offset=0 if phase=='development' else 1000000
    # Main fitting receives train+validation; main evaluation receives test alone.
    spans=[(0,3200,0),(3200,4000,1),(4000,6000,2)] if n==6000 else ([(0,3200,0),(3200,4000,1)] if n==4000 else [(0,n,partition_start)])
    for start,stop,part in spans:
        rr=np.random.default_rng(470000000+offset+seed*1000+step*10+part)
        if slots>=2:
            for fam in range(start,stop,2):shuffled[fam:fam+2]=raw[fam:fam+2][:,rr.permutation(slots)]
        group=raw[start:stop].reshape(-1,2,slots,36)
        shift=int(rr.integers(1,len(group)))
        donors=(np.arange(len(group))+shift)%len(group)
        assert np.all(donors!=np.arange(len(group)))
        mismatched[start:stop]=group[donors].reshape(-1,slots,36)
    np.testing.assert_allclose(shuffled.sum(axis=1),raw.sum(axis=1),rtol=1e-12,atol=1e-12)
    views={'current':current,'current_rff':np.concatenate([current,rff],axis=1),
        'history':np.concatenate([current,history],axis=1),'shuffled':np.concatenate([current,shuffled.reshape(n,-1)],axis=1),
        'mismatched':np.concatenate([current,mismatched.reshape(n,-1)],axis=1)}
    for name,x in views.items():np.testing.assert_array_equal(x[:,:388],current)
    return views


def arm_view(arm):return arm.rsplit('_',1)[0]


def natural_targets(probs,labels):
    truth=probs.astype(np.float64)[np.arange(len(labels))[:,None],np.arange(7)[None,:],labels[:,None]]
    nll=-np.log(np.clip(truth,1e-8,1))
    return nll[:,STEPS]-nll[:,-1,None]


def paired_gaussian(rng):
    return ((rng.normal(size=(3000,1))+rng.normal(size=(3000,2)))/np.sqrt(2)).reshape(6000)
