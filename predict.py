"""
predict.py — предсказание по обученной модели (БЕЗ обучения).

Запуск:  python predict.py --test hard_test.csv              # итоговая смесь A (2/3) + B (1/3) -> submission_seed_{SEED}.csv
         python predict.py --test hard_test.csv --members a  # только ансамбль A -> submission_seed_{SEED}_A.csv
Значения клипуются в [0, 100]. SEED по умолчанию берётся из meta.json (тот, с которым обучены веса).
"""
import argparse, json
import numpy as np, pandas as pd, lightgbm as lgb
from train import make_frame

def ensemble(prefix, X, n):
    return np.mean([lgb.Booster(model_file=f'models/{prefix}_{i:02d}.txt').predict(X) for i in range(n)], axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--test', default='hard_test.csv')
    ap.add_argument('--meta', default='meta.json')
    ap.add_argument('--members', choices=['blend', 'a', 'b'], default='blend')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--outdir', default='.')
    a = ap.parse_args()

    meta = json.load(open(a.meta))
    seed = a.seed if a.seed is not None else meta['seed']
    te = pd.read_csv(a.test)
    X, _ = make_frame(te, meta['cat_maps'])
    X = X[meta['columns']]
    pa = ensemble('a', X, meta['n_seeds']) if a.members in ('blend', 'a') else None
    pb = ensemble('b', X, meta['n_seeds']) if a.members in ('blend', 'b') else None
    p = {'a': pa, 'b': pb}.get(a.members)
    if a.members == 'blend':
        p = (1.0 - meta['w_b']) * pa + meta['w_b'] * pb
    p = np.clip(p, 0.0, 100.0)

    suffix = '' if a.members == 'blend' else '_' + a.members.upper()
    path = f'{a.outdir}/submission_seed_{seed}{suffix}.csv'
    pd.DataFrame({'customer_id': te['customer_id'], 'protection_score': np.round(p, 4)}).to_csv(path, index=False)
    print('saved', path, '| members =', a.members, '| mean=%.2f std=%.2f min=%.2f max=%.2f' % (p.mean(), p.std(), p.min(), p.max()))

if __name__ == '__main__':
    main()
