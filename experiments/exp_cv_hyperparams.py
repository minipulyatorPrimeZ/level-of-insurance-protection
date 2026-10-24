"""
exp_cv_hyperparams.py — гипотезы о гиперпараметрах LightGBM и о смеси двух ансамблей, проверка ТОЛЬКО на train (повторная K-fold CV).

Идея: метки train сильно зашумлены (RMSE любой модели ≈ 10), поэтому абсолютный CV-RMSE не информативен. Но если шум
не зависит от признаков, то РАЗНОСТЬ ошибок двух моделей на одних и тех же объектах несёт информацию о том, какая модель
ближе к истинной зависимости. Поэтому сравниваем варианты парно с финальной конфигурацией A: mean(err_variant - err_A) ± SE.
Положительная разность = вариант хуже A.

Смесь проверяется честно: «A (2/3) + B (1/3)» сравнивается не с одиночной A, а с «A (2/3) + A' (1/3)», где A' — та же
конфигурация A с другим сидом. Так отделяется эффект разнообразия конфигураций от простого усреднения двух моделей.

Запуск: python exp_cv_hyperparams.py --data_dir ..  (≈ 5 мин на 2 ядрах)
"""
import argparse, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
sys.path.insert(0, '..')
from train import make_frame, PARAMS_A, PARAMS_B, W_B

VARIANTS = {
    'A (финал: colsample 1.0, mcs 20)': {},
    "A' (то же, другой сид)":      dict(_seed_shift=1000),
    'B (colsample 0.6, subsample 0.7, mcs 40)': {k: PARAMS_B[k] for k in ('colsample_bytree', 'subsample', 'min_child_samples')},
    'colsample 0.6':   dict(colsample_bytree=0.6),
    'colsample 0.4':   dict(colsample_bytree=0.4),
    'leaves 7':        dict(num_leaves=7),
    'leaves 31':       dict(num_leaves=31),
    'leaves 63':       dict(num_leaves=63),
    'trees 350':       dict(n_estimators=350),
    'trees 1400':      dict(n_estimators=1400),
    'min_child 40':    dict(min_child_samples=40),
    'min_child 80':    dict(min_child_samples=80),
    'lambda 3':        dict(reg_lambda=3),
    'lambda 30':       dict(reg_lambda=30),
}
KA, KA2, KB = list(VARIANTS)[:3]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data_dir', default='..'); ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--seed', type=int, default=2026); a = ap.parse_args()
    tr = pd.read_csv(f'{a.data_dir}/hard_train.csv'); y = tr.protection_score.values
    X, _ = make_frame(tr)
    err = {k: np.zeros(len(y)) for k in list(VARIANTS) + ['СМЕСЬ 2/3 A + 1/3 B', "КОНТРОЛЬ 2/3 A + 1/3 A'"]}
    t0 = time.time()
    for r in range(a.repeats):
        pr = {k: np.zeros(len(y)) for k in VARIANTS}
        for ia, ib in KFold(5, shuffle=True, random_state=a.seed + r).split(X):
            for k, upd in VARIANTS.items():
                upd = dict(upd); shift = upd.pop('_seed_shift', 0)
                p = dict(PARAMS_A); p.update(upd)
                pr[k][ib] = lgb.LGBMRegressor(**p, random_state=a.seed + r + shift).fit(X.iloc[ia], y[ia]).predict(X.iloc[ib])
        for k in VARIANTS: err[k] += (pr[k] - y) ** 2 / a.repeats
        err['СМЕСЬ 2/3 A + 1/3 B'] += (((1 - W_B) * pr[KA] + W_B * pr[KB]) - y) ** 2 / a.repeats
        err["КОНТРОЛЬ 2/3 A + 1/3 A'"] += (((1 - W_B) * pr[KA] + W_B * pr[KA2]) - y) ** 2 / a.repeats
        print(f'repeat {r + 1}/{a.repeats} done, {time.time() - t0:.0f}s', flush=True)
    rows = []
    for k, e in err.items():
        d = e - err[KA]; rows.append((k, np.sqrt(e.mean()), d.mean(), d.std() / np.sqrt(len(y))))
    R = pd.DataFrame(rows, columns=['variant', 'CV_RMSE', 'dMSE_vs_A', 'SE'])
    R['z'] = R.dMSE_vs_A / R.SE.replace(0, np.nan)
    print(R.round(3).to_string(index=False)); R.to_csv('exp_cv_hyperparams_result.csv', index=False)
    d = err['СМЕСЬ 2/3 A + 1/3 B'] - err["КОНТРОЛЬ 2/3 A + 1/3 A'"]
    print(f'\nСмесь с B против контроля с A\': dMSE = {d.mean():+.3f} ± {d.std() / np.sqrt(len(y)):.3f} (минус = B даёт больше, чем второй сид A)')

if __name__ == '__main__':
    main()
