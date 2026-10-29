"""Step 2-4: train, tune, compare and evaluate three classifiers for next-day flood risk.
Models: Logistic Regression, Random Forest, Histogram Gradient Boosting (sklearn).
Split : chronological - train 2023-2024, test 2025-01-01 .. 2026-08-31 (two monsoons).
"""
import json, warnings, time
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_predict
from sklearn.metrics import (fbeta_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, brier_score_loss, matthews_corrcoef,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.inspection import permutation_importance
from sklearn.base import clone
warnings.filterwarnings('ignore')

D, O = '../data', '../outputs'
FIG = f'{O}/figs'
C = {'Logistic Regression': '#2a78d6', 'Random Forest': '#eb6834',
     'Gradient Boosting': '#1baf7a', 'Persistence baseline': '#8a8984'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#8a8984', 'axes.grid': True,
                     'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'figure.dpi': 150,
                     'axes.titleweight': 'bold', 'axes.titlesize': 11})

NUM = ['precipitation_mm', 'precipitation_hours', 'soil_moisture_0_100cm_m3m3', 'temperature_mean_c',
       'dew_point_mean_c', 'relative_humidity_mean_pct', 'wind_speed_max_kmh', 'wind_gusts_max_kmh',
       'wind_dir_sin', 'wind_dir_cos', 'rain_lag1', 'rain_lag2', 'rain_lag3', 'rain_sum_3d', 'rain_sum_7d',
       'rain_sum_14d', 'rain_sum_30d', 'rain_hours_sum_3d', 'soil_moist_change_3d', 'dewpoint_depression',
       'q_norm', 'q_norm_lag1', 'q_norm_lag2', 'q_change_1d', 'q_norm_max_7d', 'upstream_q_norm',
       'upstream_rain_3d', 'has_upstream', 'q_to_thr', 'q_to_thr_lag1', 'q_to_thr_max_3d', 'flood_today', 'doy_sin', 'doy_cos', 'is_monsoon', 'elevation_m']
CAT = ['location']
TARGET = 'flood_next_day'

df = pd.read_csv(f'{D}/nepal_features.csv', parse_dates=['date']).sort_values(['date', 'location'])
train = df[df.date <= '2024-12-31']; test = df[df.date >= '2025-01-01']
Xtr, ytr, Xte, yte = train[NUM + CAT], train[TARGET], test[NUM + CAT], test[TARGET]
print('train', Xtr.shape, ytr.mean().round(4), '| test', Xte.shape, yte.mean().round(4))

def prep(scale):
    return ColumnTransformer([('num', StandardScaler() if scale else 'passthrough', NUM),
                              ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT)])

models = {
    'Logistic Regression': (Pipeline([('prep', prep(True)), ('clf', LogisticRegression(max_iter=3000, class_weight='balanced'))]),
                            {'clf__C': [0.01, 0.1, 1, 10], 'clf__class_weight': [None, 'balanced']}),
    'Random Forest': (Pipeline([('prep', prep(False)), ('clf', RandomForestClassifier(n_estimators=400, class_weight='balanced_subsample', n_jobs=-1, random_state=42))]),
                      {'clf__max_depth': [6, 10, None], 'clf__min_samples_leaf': [2, 5, 10], 'clf__class_weight': [None, 'balanced_subsample']}),
    'Gradient Boosting': (Pipeline([('prep', prep(False)), ('clf', HistGradientBoostingClassifier(class_weight='balanced', random_state=42, early_stopping=False))]),
                          {'clf__learning_rate': [0.03, 0.1], 'clf__max_depth': [3, 6], 'clf__max_iter': [200, 400], 'clf__l2_regularization': [0.0, 1.0], 'clf__class_weight': [None, 'balanced']}),
}
# Expanding-window, forward-chaining folds (never validate on the past).
# Calendar-based so every training window contains a full monsoon.
class SeasonSplit:
    cuts = [('2023-12-31', '2024-08-31'), ('2024-08-31', '2024-12-31')]
    def split(self, X, y=None, groups=None):
        d = train['date'].values
        for a, b in self.cuts:
            yield np.where(d <= np.datetime64(a))[0], np.where((d > np.datetime64(a)) & (d <= np.datetime64(b)))[0]
    def get_n_splits(self, *a, **k): return len(self.cuts)
tscv = SeasonSplit()
results, probs, fitted, cvinfo = {}, {}, {}, {}

def metrics(y, p, thr):
    yhat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {'threshold': round(thr, 3), 'precision': precision_score(y, yhat, zero_division=0),
            'recall': recall_score(y, yhat), 'f1': f1_score(y, yhat), 'f2': fbeta_score(y, yhat, beta=2), 'mcc': matthews_corrcoef(y, yhat),
            'roc_auc': roc_auc_score(y, p), 'pr_auc': average_precision_score(y, p),
            'brier': brier_score_loss(y, p), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)}

for name, (pipe, grid) in models.items():
    t0 = time.time()
    gs = GridSearchCV(pipe, grid, scoring='average_precision', cv=tscv, n_jobs=-1, error_score='raise').fit(Xtr, ytr)
    best = gs.best_estimator_
    # decision threshold chosen on out-of-fold TRAIN predictions (maximise F1), never on test
    oof_idx = np.concatenate([te for _, te in tscv.split(Xtr)])
    oof = np.zeros(len(Xtr));
    for tr_i, te_i in tscv.split(Xtr):
        m = clone(best).fit(Xtr.iloc[tr_i], ytr.iloc[tr_i]); oof[te_i] = m.predict_proba(Xtr.iloc[te_i])[:, 1]
    pr, rc, th = precision_recall_curve(ytr.iloc[oof_idx], oof[oof_idx])
    # F2 weights recall 2x precision: a missed flood costs far more than a false alarm
    f2s = 5 * pr * rc / np.clip(4 * pr + rc, 1e-9, None)
    thr = float(th[np.nanargmax(f2s[:-1])])
    p = best.predict_proba(Xte)[:, 1]
    results[name] = metrics(yte, p, thr); probs[name] = p; fitted[name] = best
    results[name]['train_seconds'] = round(time.time() - t0, 1)
    cvinfo[name] = {'best_params': gs.best_params_, 'cv_pr_auc': round(gs.best_score_, 4)}
    print(name, cvinfo[name], {k: round(v, 3) if isinstance(v, float) else v for k, v in results[name].items()})

# Persistence baseline: "tomorrow looks like today"
pb = test['flood_today'].values.astype(float)
results['Persistence baseline'] = metrics(yte, pb, 0.5)

# Onset events (no flood today, flood tomorrow) - the hard, valuable cases
onset = (test['flood_today'].values == 0) & (yte.values == 1)
for name in models:
    yhat = probs[name] >= results[name]['threshold']
    results[name]['onset_recall'] = float(yhat[onset].mean())
results['Persistence baseline']['onset_recall'] = 0.0
print('onset events in test:', int(onset.sum()))

res = pd.DataFrame(results).T
res.to_csv(f'{O}/model_comparison.csv'); print(res.round(3).to_string())

# Best model = highest PR-AUC (robust for 4-5% positive class); tie-break F1
best_name = max(models, key=lambda n: (round(results[n]['pr_auc'], 3), results[n]['f2']))
print('BEST', best_name)

# ---------------- Figures ----------------
# 1 ROC + PR curves
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
for n in models:
    f, t, _ = roc_curve(yte, probs[n]); ax[0].plot(f, t, lw=2, color=C[n], label=f"{n} (AUC {results[n]['roc_auc']:.3f})")
    pr, rc, _ = precision_recall_curve(yte, probs[n]); ax[1].plot(rc, pr, lw=2, color=C[n], label=f"{n} (AP {results[n]['pr_auc']:.3f})")
ax[0].plot([0, 1], [0, 1], ls='--', color='#8a8984', lw=1)
ax[1].axhline(yte.mean(), ls='--', color='#8a8984', lw=1, label=f'No-skill ({yte.mean():.3f})')
ax[0].set(xlabel='False positive rate', ylabel='True positive rate', title='ROC curve (test 2025-26)')
ax[1].set(xlabel='Recall', ylabel='Precision', title='Precision-recall curve (test 2025-26)')
for a in ax: a.legend(frameon=False, fontsize=8, loc='lower right' if a is ax[0] else 'lower left')
plt.tight_layout(); plt.savefig(f'{FIG}/roc_pr.png'); plt.close()

# 2 Metric bars
mets = ['precision', 'recall', 'f1', 'f2', 'mcc', 'pr_auc', 'onset_recall']
labels = ['Precision', 'Recall', 'F1', 'F2', 'MCC', 'PR-AUC', 'Onset recall']
names = list(models) + ['Persistence baseline']
fig, ax = plt.subplots(figsize=(10, 4)); w = 0.2; x = np.arange(len(mets))
for i, n in enumerate(names):
    v = [results[n][m] for m in mets]
    b = ax.bar(x + (i - 1.5) * w, v, w * 0.9, color=C[n], label=n)
    for xi, vi in zip(x + (i - 1.5) * w, v): ax.text(xi, vi + 0.01, f'{vi:.2f}', ha='center', fontsize=6.5, color='#52514e')
ax.set_xticks(x, labels); ax.set_ylim(0, 1.08); ax.set_ylabel('Score (higher is better)')
ax.set_title('Test-set performance by model'); ax.legend(frameon=False, ncol=4, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.1))
plt.tight_layout(); plt.savefig(f'{FIG}/metric_bars.png'); plt.close()

# 3 Confusion matrices
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.3))
for a, n in zip(ax, models):
    r = results[n]; cm = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
    a.imshow(cm, cmap='Blues', vmin=0, vmax=cm[1].sum() * 3); a.grid(False)
    for i in range(2):
        for j in range(2): a.text(j, i, cm[i, j], ha='center', va='center', fontsize=12, weight='bold', color='#0b0b0b')
    a.set_xticks([0, 1], ['No flood', 'Flood']); a.set_yticks([0, 1], ['No flood', 'Flood'])
    a.set_xlabel('Predicted'); a.set_ylabel('Actual'); a.set_title(n, fontsize=10)
plt.tight_layout(); plt.savefig(f'{FIG}/confusion.png'); plt.close()

# 4 Permutation importance for best model
bm = fitted[best_name]
pi = permutation_importance(bm, Xte, yte, scoring='average_precision', n_repeats=8, random_state=42, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=NUM + CAT).sort_values().tail(15)
imp.to_csv(f'{O}/feature_importance.csv')
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.barh(imp.index, imp.values, color='#2a78d6', height=0.7); ax.set_xlabel('Drop in PR-AUC when feature is shuffled')
ax.set_title(f'Top 15 features - {best_name}'); ax.grid(axis='y', visible=False)
plt.tight_layout(); plt.savefig(f'{FIG}/feature_importance.png'); plt.close()

# 5 Timeline example: station with most test floods
st = test.assign(p=probs[best_name]).groupby('location')[TARGET].sum().idxmax()
s = test.assign(p=probs[best_name])[lambda d: d.location == st]
s = s[(s.date >= '2025-05-01') & (s.date <= '2025-11-30')]
fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True, gridspec_kw={'height_ratios': [1.3, 1]})
ax[0].plot(s.date, s.q_next, color='#2a78d6', lw=1.6, label='Next-day discharge')
ax[0].axhline(s.threshold_m3s.iloc[0], color='#d03b3b', ls='--', lw=1.2, label='Flood threshold (P95)')
ax[0].set_ylabel('m³/s'); ax[0].legend(frameon=False, fontsize=8); ax[0].set_title(f'{st}: 2025 monsoon - observed vs predicted risk')
ax[1].fill_between(s.date, s.p, color='#1baf7a', alpha=0.35, lw=0); ax[1].plot(s.date, s.p, color='#1baf7a', lw=1.4, label='Predicted flood probability')
ax[1].axhline(results[best_name]['threshold'], color='#52514e', ls=':', lw=1.2, label='Alert threshold')
ax[1].set_ylabel('Probability'); ax[1].set_ylim(0, 1.02); ax[1].legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f'{FIG}/timeline.png'); plt.close()

# Final deployment model: refit best config on all data 2023-2026 (threshold kept from train OOF)
from sklearn.base import clone
final = clone(bm).fit(df[NUM + CAT], df[TARGET])
joblib.dump({'model': final, 'model_eval': bm, 'features': NUM + CAT, 'num': NUM, 'threshold': results[best_name]['threshold'],
             'name': best_name}, f'{O}/flood_model.joblib')
json.dump({'results': results, 'cv': cvinfo, 'best': best_name, 'onset_events': int(onset.sum()),
           'n_train': len(train), 'n_test': len(test), 'pos_train': int(ytr.sum()), 'pos_test': int(yte.sum()),
           'timeline_station': st}, open(f'{O}/results.json', 'w'), indent=2, default=float)
