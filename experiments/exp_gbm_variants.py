"""
exp_gbm_variants.py - гипотезы 5–11: варианты решений из чистого бустинга.
Запуск: python exp_gbm_variants.py --data_dir .. --test ../hard_test.csv --outdir variants --n_seeds 5
"""
import argparse, os, sys
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostRegressor          # pip install catboost (только для гипотезы 5)
sys.path.insert(0, '..')
from train import make_frame, PARAMS
OLD = dict(min_child_samples=40)

VARIANTS = {
    'L0_colsample06_subsample07': dict(colsample_bytree=0.6, subsample=0.7),
    'L4_allcols':                 {},
    'colsample04':                dict(colsample_bytree=0.4),
    'leaves7_trees1500':          dict(num_leaves=7, n_estimators=1500),
    'leaves31':                   dict(num_leaves=31, min_child_samples=20, reg_lambda=5),
    'leaves63':                   dict(num_leaves=63, min_child_samples=10, reg_lambda=1, n_estimators=500),
    'trees1400':                  dict(n_estimators=1400),
    'min_child20':                dict(min_child_samples=20),     # = финальное значение A
    'min_child80':                dict(min_child_samples=80),
    'lambda30':                   dict(reg_lambda=30),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', default='..'); ap.add_argument('--test', default='../hard_test.csv')
    ap.add_argument('--outdir', default='variants'); ap.add_argument('--n_seeds', type=int, default=5); ap.add_argument('--seed', type=int, default=679035307)
    ap.add_argument('--with_catboost', action='store_true'); a = ap.parse_args()
    tr = pd.read_csv(f'{a.data_dir}/hard_train.csv'); te = pd.read_csv(a.test); y = tr.protection_score.values
    X, cm = make_frame(tr); Xt, _ = make_frame(te, cm); Xt = Xt[X.columns]
    os.makedirs(a.outdir, exist_ok=True)
    save = lambda name, p: pd.DataFrame({'customer_id': te.customer_id, 'protection_score': np.round(np.clip(p, 0, 100), 4)}).to_csv(f'{a.outdir}/sub_{name}.csv', index=False)
    for name, upd in VARIANTS.items():
        print(name, upd)
        p = dict(PARAMS); p.update(OLD); p.update(upd)
        save(name, np.mean([lgb.LGBMRegressor(**p, random_state=a.seed + i).fit(X, y).predict(Xt) for i in range(a.n_seeds)], 0)); print(name, flush=True)
    if a.with_catboost:   # гипотеза 5: CatBoost вместо LightGBM
        save('catboost_d6', np.mean([CatBoostRegressor(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5, verbose=0,
                                     random_seed=a.seed + i, thread_count=2).fit(X, y).predict(Xt) for i in range(3)], 0)); print('catboost_d6')

if __name__ == '__main__':
    main()
