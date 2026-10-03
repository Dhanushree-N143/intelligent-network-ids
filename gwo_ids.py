import argparse, os, re, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

CATEGORICAL = ['protocol_type','service','flag']
FEATURE_NAMES = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate','dst_host_srv_rerror_rate']

def load_arff(path):
    attrs=[]; rows=[]; data=False
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            line=raw.strip()
            if not line or line.startswith('%'): continue
            if not data:
                if line.lower().startswith('@attribute'):
                    m=re.match(r"@attribute\s+(?:'([^']+)'|([^\s]+))\s+(.+)$", line, re.I)
                    if m: attrs.append(m.group(1) or m.group(2))
                elif line.lower()=='@data': data=True
            else:
                vals=[]; cur=''; quote=None
                for ch in line:
                    if ch in "'\"":
                        if quote is None: quote=ch
                        elif quote==ch: quote=None
                        else: cur+=ch
                    elif ch==',' and quote is None:
                        vals.append(cur.strip()); cur=''
                    else: cur+=ch
                vals.append(cur.strip())
                if len(vals)==len(attrs): rows.append(vals)
    return pd.DataFrame(rows, columns=attrs)

def prepare(train_path, test_path, sample=None, seed=42, progress=None):
    if progress: progress('loading','Loading dataset...',0.02)
    tr=load_arff(train_path); te=load_arff(test_path)
    if progress: progress('preprocessing','Preprocessing...',0.10)
    target=tr.columns[-1]
    # normalize accidental whitespace in categorical values
    for c in CATEGORICAL:
        tr[c]=tr[c].astype(str).str.strip().str.strip("'\"")
        te[c]=te[c].astype(str).str.strip().str.strip("'\"")
    tr=tr.drop_duplicates().reset_index(drop=True)
    te=te.drop_duplicates().reset_index(drop=True)
    if sample and sample < len(tr):
        tr,_=train_test_split(tr, train_size=sample, stratify=tr[target], random_state=seed)
        tr=tr.reset_index(drop=True)
    ytr=(tr[target].astype(str).str.strip().str.lower()!='normal').astype(int).to_numpy()
    yte=(te[target].astype(str).str.strip().str.lower()!='normal').astype(int).to_numpy()
    cols=[c for c in tr.columns if c!=target]
    Xtr=tr[cols].copy(); Xte=te[cols].copy()
    for c in cols:
        if c in CATEGORICAL:
            combined=pd.concat([Xtr[c],Xte[c]],ignore_index=True).astype(str)
            cats={v:i for i,v in enumerate(sorted(combined.unique()))}
            Xtr[c]=Xtr[c].map(cats); Xte[c]=Xte[c].map(cats)
        else:
            Xtr[c]=pd.to_numeric(Xtr[c],errors='coerce'); Xte[c]=pd.to_numeric(Xte[c],errors='coerce')
    Xtr=Xtr.replace([np.inf,-np.inf],np.nan); Xte=Xte.replace([np.inf,-np.inf],np.nan)
    med=Xtr.median(numeric_only=True)
    Xtr=Xtr.fillna(med).fillna(0); Xte=Xte.fillna(med).fillna(0)
    scaler=MinMaxScaler(); Xtr=scaler.fit_transform(Xtr); Xte=scaler.transform(Xte)
    return Xtr,ytr,Xte,yte,cols,len(tr),len(te)

class BinaryGWO:
    def __init__(self, n_features, pop=5, iters=10, alpha=0.99, beta=0.01, seed=42, rf_trees=50, callback=None):
        self.callback=callback
        self.d=n_features; self.pop=pop; self.iters=iters; self.alpha=alpha; self.beta=beta; self.rng=np.random.default_rng(seed); self.rf_trees=rf_trees
        self.cache={}; self.history=[]; self.evals=0
    def _mask(self, pos):
        mask=pos>=0.5
        if not mask.any(): mask[np.argmax(pos)]=True
        return mask
    def _fitness(self, pos, X, y):
        mask=self._mask(pos); key=np.packbits(mask).tobytes()
        if key in self.cache: return self.cache[key]
        idx=np.arange(len(y)); a,b=train_test_split(idx,test_size=0.2,stratify=y,random_state=42)
        clf=RandomForestClassifier(n_estimators=self.rf_trees,random_state=42,n_jobs=-1,class_weight='balanced_subsample',max_depth=None)
        clf.fit(X[a][:,mask],y[a]); pred=clf.predict(X[b][:,mask]); acc=accuracy_score(y[b],pred)
        fit=self.alpha*(1-acc)+self.beta*(mask.sum()/self.d)
        self.cache[key]=(fit,mask.copy()); self.evals+=1
        return self.cache[key]
    def optimize(self,X,y):
        wolves=self.rng.random((self.pop,self.d)); scores=np.empty(self.pop); masks=[]
        print(f'Initial population: evaluating {self.pop} candidates with RF validation...')
        for i in range(self.pop):
            scores[i],m=self._fitness(wolves[i],X,y); masks.append(m)
            print(f'  Initial candidate {i+1}/{self.pop} | Fitness={scores[i]:.6f} | Selected={m.sum()}/{self.d}',flush=True)
            if self.callback: self.callback('init',i+1,self.pop,float(scores[i]),int(m.sum()))
        order=np.argsort(scores); alpha=wolves[order[0]].copy(); beta=wolves[order[1 if self.pop>1 else 0]].copy(); delta=wolves[order[2 if self.pop>2 else 0]].copy(); best=scores[order[0]]; bestmask=masks[order[0]].copy()
        for t in range(self.iters):
            a=2-2*t/max(1,self.iters-1)
            new=np.empty_like(wolves)
            for i in range(self.pop):
                r1=self.rng.random(self.d); r2=self.rng.random(self.d); A1=2*a*r1-a; C1=2*r2; D1=np.abs(C1*alpha-wolves[i]); X1=alpha-A1*D1
                r1=self.rng.random(self.d); r2=self.rng.random(self.d); A2=2*a*r1-a; C2=2*r2; D2=np.abs(C2*beta-wolves[i]); X2=beta-A2*D2
                r1=self.rng.random(self.d); r2=self.rng.random(self.d); A3=2*a*r1-a; C3=2*r2; D3=np.abs(C3*delta-wolves[i]); X3=delta-A3*D3
                new[i]=1/(1+np.exp(-np.clip((X1+X2+X3)/3,-20,20)))
            wolves=new
            for i in range(self.pop):
                sc,m=self._fitness(wolves[i],X,y)
                if sc<best: best=sc; bestmask=m.copy(); alpha=wolves[i].copy()
            # refresh alpha/beta/delta from current population scores
            cur=[]; curm=[]
            for i in range(self.pop):
                sc,m=self._fitness(wolves[i],X,y); cur.append(sc); curm.append(m)
            order=np.argsort(cur); alpha=wolves[order[0]].copy(); beta=wolves[order[1 if self.pop>1 else 0]].copy(); delta=wolves[order[2 if self.pop>2 else 0]].copy()
            if cur[order[0]]<best: best=cur[order[0]]; bestmask=curm[order[0]].copy()
            self.history.append((t+1,best,int(bestmask.sum())))
            print(f'Iteration {t+1:3d}/{self.iters} | Fitness={best:.6f} | Selected={bestmask.sum():2d}/{self.d} | Unique fitness evaluations={self.evals}',flush=True)
            if self.callback: self.callback('iteration',t+1,self.iters,float(best),int(bestmask.sum()))
        return best,bestmask

def _noop(*a, **k): pass

def _save_run_extras(results_dir, prefix, mode, names, mask, final, yte, pred, ntr, nte, params):
    """Additive outputs used by the dashboard (predictions, importances, metadata)."""
    import json, datetime
    pd.DataFrame({'actual': yte, 'predicted': pred}).to_csv(
        os.path.join(results_dir, f'{prefix}_predictions.csv'), index=False)
    sel_names = [n for n, m in zip(names, mask) if m]
    pd.DataFrame({'feature': sel_names, 'importance': final.feature_importances_}).to_csv(
        os.path.join(results_dir, f'{prefix}_feature_importance.csv'), index=False)
    import sklearn
    meta = {'mode': mode, 'library_versions': {'scikit-learn': sklearn.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}, 'training_rows': int(ntr), 'test_rows': int(nte),
            'total_features': int(len(names)), 'selected_features': int(mask.sum()),
            'parameters': params,
            'timestamp': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(os.path.join(results_dir, f'{prefix}_run_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

def run_baseline(train_arff, test_arff, sample=5000, trees=50, seed=42,
                 results_dir='results', progress=None):
    """All-feature Random Forest baseline (no GWO). Same logic as `--baseline`."""
    progress = progress or _noop
    t0 = time.time()
    X, y, Xte, yte, names, ntr, nte = prepare(train_arff, test_arff, sample, seed, progress)

    print('=' * 72)
    print('RANDOM FOREST — ALL-FEATURE BASELINE')
    print('=' * 72)
    print(f'Training rows      : {ntr}')
    print(f'Test rows          : {nte}')
    print(f'Features           : {X.shape[1]}')
    print(f'Classes            : {len(np.unique(y))}')
    print(f'RF trees           : {trees}')
    print()

    os.makedirs(results_dir, exist_ok=True)
    # True baseline: use all available features, with no GWO.
    mask = np.ones(X.shape[1], dtype=bool)

    progress('training', 'Training Random Forest on all features...', 0.40)
    final = RandomForestClassifier(
        n_estimators=max(trees, 100),
        random_state=seed,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    final.fit(X, y)
    progress('evaluating', 'Evaluating test set...', 0.80)
    pred = final.predict(Xte)

    acc = accuracy_score(yte, pred)
    pre = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    runtime = time.time() - t0

    progress('saving', 'Saving results...', 0.92)
    pd.DataFrame({'feature': names, 'selected': mask.astype(int)}).to_csv(
        os.path.join(results_dir, 'baseline_selected_features.csv'), index=False)

    result = {
        'selected_features': int(mask.sum()),
        'feature_reduction_percent': 0.0,
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'runtime_seconds': runtime
    }
    pd.DataFrame([result]).to_csv(os.path.join(results_dir, 'baseline_results.csv'), index=False)
    _save_run_extras(results_dir, 'baseline', 'baseline', names, mask, final, yte, pred, ntr, nte,
                     {'sample': sample, 'trees': trees, 'seed': seed})

    print('Run result — ALL-FEATURE RANDOM FOREST BASELINE:')
    print(f'Features         : {mask.sum()}/{len(mask)}')
    print('Reduction        : 0.00%')
    print(f'Accuracy         : {acc:.4f}')
    print(f'Precision        : {pre:.4f}')
    print(f'Recall           : {rec:.4f}')
    print(f'F1-score         : {f1:.4f}')
    print(f'Runtime          : {runtime:.2f} s')
    print('\nResults saved to results/')
    progress('done', 'Done.', 1.0)
    return result

def run_gwo(train_arff, test_arff, sample=5000, pop=5, iters=10, trees=50, seed=42,
            results_dir='results', progress=None):
    """Existing GWO + Random Forest IDS. Same logic as the original CLI run."""
    progress = progress or _noop
    t0 = time.time()
    X, y, Xte, yte, names, ntr, nte = prepare(train_arff, test_arff, sample, seed, progress)

    print('=' * 72)
    print('GWO + RANDOM FOREST — EXISTING IDS BASELINE')
    print('=' * 72)
    print(f'Training rows      : {ntr}')
    print(f'Test rows          : {nte}')
    print(f'Features           : {X.shape[1]}')
    print(f'Classes            : {len(np.unique(y))}')
    print(f'Population         : {pop}')
    print(f'Iterations         : {iters}')
    print('Fitness validation : 80/20 stratified holdout')
    print(f'RF trees           : {trees}')
    print()

    os.makedirs(results_dir, exist_ok=True)

    def gwo_cb(event, cur, total, fit, nsel):
        if event == 'init':
            progress('init', f'Evaluating population... candidate {cur}/{total} '
                     f'(fitness {fit:.6f}, {nsel} features)', 0.15 + 0.15 * cur / total)
        else:
            progress('iteration', f'Running iterations... {cur}/{total} '
                     f'(best fitness {fit:.6f}, {nsel} features)', 0.30 + 0.50 * cur / total)

    progress('init', 'Initializing GWO...', 0.15)
    gwo = BinaryGWO(X.shape[1], pop, iters, seed=seed, rf_trees=trees, callback=gwo_cb)
    best, mask = gwo.optimize(X, y)

    progress('training', 'Training Random Forest on selected features...', 0.82)
    final = RandomForestClassifier(
        n_estimators=max(trees, 100),
        random_state=seed,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    final.fit(X[:, mask], y)
    progress('evaluating', 'Evaluating test set...', 0.90)
    pred = final.predict(Xte[:, mask])

    acc = accuracy_score(yte, pred)
    pre = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)

    progress('saving', 'Saving results...', 0.95)
    pd.DataFrame(gwo.history, columns=['iteration', 'fitness', 'selected_features']).to_csv(
        os.path.join(results_dir, 'gwo_convergence.csv'), index=False)
    pd.DataFrame({'feature': names, 'selected': mask.astype(int)}).to_csv(
        os.path.join(results_dir, 'gwo_selected_features.csv'), index=False)

    result = {
        'fitness': best,
        'selected_features': int(mask.sum()),
        'feature_reduction_percent': 100 * (1 - mask.sum() / len(mask)),
        'accuracy': acc,
        'precision': pre,
        'recall': rec,
        'f1': f1,
        'runtime_seconds': time.time() - t0,
        'unique_fitness_evaluations': gwo.evals
    }
    pd.DataFrame([result]).to_csv(os.path.join(results_dir, 'gwo_results.csv'), index=False)
    _save_run_extras(results_dir, 'gwo', 'gwo_rf', names, mask, final, yte, pred, ntr, nte,
                     {'sample': sample, 'pop': pop, 'iters': iters, 'trees': trees, 'seed': seed})

    print('\nRun result:')
    print(f'Fitness          : {best:.6f}')
    print(f'Selected features: {mask.sum()}')
    print(f'Reduction        : {100 * (1 - mask.sum() / len(mask)):.2f}%')
    print(f'Accuracy         : {acc:.4f}')
    print(f'Precision        : {pre:.4f}')
    print(f'Recall           : {rec:.4f}')
    print(f'F1-score         : {f1:.4f}')
    print(f'Runtime          : {time.time() - t0:.2f} s')
    print(f'Unique evaluations: {gwo.evals}')
    print('\nResults saved to results/')
    progress('done', 'Done.', 1.0)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-arff', required=True)
    ap.add_argument('--test-arff', required=True)
    ap.add_argument('--sample', type=int, default=5000)
    ap.add_argument('--pop', type=int, default=5)
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--trees', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument(
        '--baseline',
        action='store_true',
        help='Run Random Forest using all features and skip GWO'
    )
    args = ap.parse_args()

    if args.baseline:
        run_baseline(args.train_arff, args.test_arff, args.sample, args.trees, args.seed)
    else:
        run_gwo(args.train_arff, args.test_arff, args.sample, args.pop, args.iters,
                args.trees, args.seed)

if __name__=='__main__': main()
