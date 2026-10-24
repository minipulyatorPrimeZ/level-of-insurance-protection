"""
train.py — обучение финальной модели (D. PRO: Клиент под защитой).

Модель: взвешенное среднее двух ансамблей LightGBM-регрессоров (шкала цели — исходная, 0..100).
  Ансамбль A (вес 2/3): все признаки в каждом дереве (colsample_bytree=1.0), subsample=0.8, min_child_samples=20.
  Ансамбль B (вес 1/3): colsample_bytree=0.6, subsample=0.7, min_child_samples=40.
  Внутри каждого ансамбля — N_SEEDS=30 моделей с разными сидами (SEED + i), простое среднее.
Используется ТОЛЬКО hard_train.csv. Тестовые данные не читаются.

Запуск:  python train.py --data_dir . --seed 679035307
Выход:   models/a_00..29.txt, models/b_00..29.txt (веса), meta.json (признаки, кодирование категорий, параметры, веса смеси, seed)
Воспроизводимость: сиды = SEED + i, deterministic=True, force_row_wise=True.
"""
import argparse, json, os, random
import numpy as np, pandas as pd, lightgbm as lgb

CATS = ['region', 'city_type', 'gender', 'education', 'family_status', 'employment', 'wealth_segment']
FILL0 = ['house_value', 'car_value', 'average_claim_cost']   # NaN = «нет объекта» -> 0 + бинарный флаг *_na
N_SEEDS = 30
W_B = 1.0 / 3.0                                               # вес ансамбля B в итоговой смеси

# Конфигурации LightGBM. Альтернативы, которые проверялись и были отвергнуты (линейная модель на logit-шкале, CatBoost,
# удаление выбросов, трансформер, другие значения гиперпараметров), реализованы скриптами в experiments/ и описаны в README.md;
# в финальное обучение они не входят.
PARAMS_A = dict(
    n_estimators=700, learning_rate=0.02, num_leaves=15, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=1.0, reg_lambda=10,
    verbose=-1, n_jobs=2, deterministic=True, force_row_wise=True,
)
PARAMS_B = dict(PARAMS_A, min_child_samples=40, subsample=0.7, colsample_bytree=0.6)
PARAMS = PARAMS_A                                             # алиас для скриптов из experiments/

def make_frame(df, cat_maps=None):
    """Единая предобработка для train и test. Категории кодируются по словарю, найденному на train."""
    X = df.drop(columns=['customer_id', 'protection_score'], errors='ignore').copy()
    for c in FILL0:
        X[c + '_na'] = X[c].isna().astype(int)
        X[c] = X[c].fillna(0)
    if cat_maps is None:
        cat_maps = {c: {v: i for i, v in enumerate(sorted(X[c].unique()))} for c in CATS}
    for c in CATS:
        X[c] = X[c].map(cat_maps[c]).fillna(-1).astype(int)
    return X, cat_maps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='.')
    ap.add_argument('--seed', type=int, default=random.SystemRandom().randint(1, 10**9))
    ap.add_argument('--out_dir', default='.')
    a = ap.parse_args()
    seed = a.seed
    random.seed(seed); np.random.seed(seed)
    print('SEED =', seed)

    tr = pd.read_csv(f'{a.data_dir}/hard_train.csv')
    y = tr['protection_score'].values
    X, cat_maps = make_frame(tr)

    os.makedirs(f'{a.out_dir}/models', exist_ok=True)
    for name, params in (('a', PARAMS_A), ('b', PARAMS_B)):
        for i in range(N_SEEDS):
            m = lgb.LGBMRegressor(**params, random_state=seed + i).fit(X, y)
            m.booster_.save_model(f'{a.out_dir}/models/{name}_{i:02d}.txt')
    json.dump(dict(seed=seed, n_seeds=N_SEEDS, w_b=W_B, columns=list(X.columns), cat_maps=cat_maps,
                   params_a=PARAMS_A, params_b=PARAMS_B), open(f'{a.out_dir}/meta.json', 'w'), indent=1, ensure_ascii=False)
    print(f'saved {2 * N_SEEDS} models -> {a.out_dir}/models, meta -> {a.out_dir}/meta.json')

if __name__ == '__main__':
    main()
