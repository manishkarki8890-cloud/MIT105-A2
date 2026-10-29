"""EDA / data-quality figures and the ML architecture diagram."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
FIG = '../outputs/figs'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#8a8984', 'axes.grid': True,
                     'grid.color': '#e6e5e1', 'grid.linewidth': 0.6, 'figure.dpi': 150,
                     'axes.titleweight': 'bold', 'axes.titlesize': 11})
raw = pd.read_csv('../data/nepal_raw.csv'); raw.columns = [c.strip() for c in raw.columns]
raw['date'] = pd.to_datetime(raw.date, format='%d-%m-%Y')
cl = pd.read_csv('../data/nepal_clean.csv', parse_dates=['date'])
ft = pd.read_csv('../data/nepal_features.csv', parse_dates=['date'])

# 1. Column-shift fix: temperature before/after
fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for a, d, t in [(ax[0], raw, 'Raw: "temperature_mean_c"'), (ax[1], cl, 'Cleaned: temperature_mean_c')]:
    m = d.groupby(d.date.dt.to_period('M'))['temperature_mean_c'].mean()
    a.plot(m.index.to_timestamp(), m.values, color='#2a78d6', lw=2)
    a.axvspan(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'), color='#eda100', alpha=0.15, lw=0)
    a.set_title(t); a.set_ylabel('Monthly mean, all stations (°C)')
ax[0].text(pd.Timestamp('2023-02-01'), 28, '2023: column\nholds soil\nmoisture', fontsize=8, color='#52514e')
plt.tight_layout(); plt.savefig(f'{FIG}/shift_fix.png'); plt.close()

# 2. Seasonality: monthly rain and flood-label rate
mon = ft.groupby(ft.date.dt.month).agg(rain=('precipitation_mm', 'mean'), flood=('flood_next_day', 'mean'))
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
mn = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
ax[0].bar(range(1, 13), mon.rain, color='#2a78d6', width=0.7); ax[0].set_title('Mean daily precipitation by month'); ax[0].set_ylabel('mm/day')
ax[1].bar(range(1, 13), mon.flood * 100, color='#eb6834', width=0.7); ax[1].set_title('Share of days labelled "flood next day"'); ax[1].set_ylabel('% of station-days')
for a in ax: a.set_xticks(range(1, 13), mn); a.grid(axis='x', visible=False)
plt.tight_layout(); plt.savefig(f'{FIG}/seasonality.png'); plt.close()

# 3. Correlation heatmap
cols = ['precipitation_mm', 'precipitation_hours', 'soil_moisture_0_100cm_m3m3', 'temperature_mean_c', 'dew_point_mean_c',
        'relative_humidity_mean_pct', 'wind_speed_max_kmh', 'rain_sum_7d', 'rain_sum_30d', 'q_norm', 'q_to_thr', 'flood_next_day']
short = ['precip', 'precip_hrs', 'soil_moist', 'temp', 'dew_pt', 'rel_hum', 'wind_max', 'rain_7d', 'rain_30d', 'q_norm', 'q_to_thr', 'FLOOD_t+1']
cm = ft[cols].corr(method='spearman').values
fig, ax = plt.subplots(figsize=(7.5, 6.3))
im = ax.imshow(cm, cmap='RdBu_r', vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(len(short)), short, rotation=45, ha='right'); ax.set_yticks(range(len(short)), short)
for i in range(len(short)):
    for j in range(len(short)):
        ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center', fontsize=6.5, color='white' if abs(cm[i, j]) > 0.6 else '#0b0b0b')
plt.colorbar(im, fraction=0.04); ax.set_title('Spearman correlation (cleaned + engineered features)')
plt.tight_layout(); plt.savefig(f'{FIG}/correlation.png'); plt.close()

# 4. Architecture diagram
fig, ax = plt.subplots(figsize=(12, 5.2)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0.75, 5.75)
def box(x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=fc, ec='#52514e', lw=1))
    ax.text(x + w / 2, y + h - 0.22, title, ha='center', va='top', weight='bold', fontsize=9.5)
    ax.text(x + w / 2, y + h - 0.55, body, ha='center', va='top', fontsize=7.6, color='#2b2b2a', linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='-|>', color='#52514e', lw=1.4))
B, O, G, Y = '#dbe8f8', '#fbe1d5', '#d4f0e4', '#fbedc8'
box(0.1, 3.5, 2.2, 2.15, '1. Data source', 'Kaggle Nepal flood-weather\n(.numbers, 13,390 rows)\n10 DHM stations\n2023-01 → 2026-08\nOpen-Meteo weather +\nGloFAS discharge', B)
box(2.65, 3.5, 2.4, 2.15, '2. Pre-processing', 'Custom .numbers parser\nFix 2023 column rotation\n(3,650 rows)\nValidity rules + interpolation\nElevation corrections\nDrop rows w/o lags (40)', B)
box(5.4, 3.5, 2.6, 2.15, '3. Feature engineering', '36 features + station:\nrain lags 1-3 d, sums 3-30 d\nsoil-moisture change\ndischarge ÷ station median\ndischarge ÷ P95 threshold\nflood_today flag\nupstream gauge signal\nseason + wind sin/cos', B)
box(8.35, 3.5, 3.55, 2.15, '4. Target', 'flood_next_day = 1 if\nQ(t+1) > station P95\n(P95 from 2023-24 only)\n\nPositive rate 4.6%\n→ imbalanced classification', Y)
box(0.1, 0.9, 3.6, 1.9, '5. Training (2023-24)', 'Forward-chaining CV\n(2 expanding folds)\nGridSearch on PR-AUC\nclass_weight searched\nAlert threshold = max F2\non out-of-fold predictions', O)
box(4.0, 0.9, 3.9, 1.9, '6. Candidate models', 'Logistic Regression (scaled)\nRandom Forest (400 trees)\nHist. Gradient Boosting\nvs. persistence baseline\n\nTest: 2025-01 → 2026-08', O)
box(8.2, 0.9, 3.7, 1.9, '7. Deployment', 'Best = Random Forest\nrefit on all data → joblib\nFlask web app + JSON API\nInputs: station, 3 days of\nweather + discharge\nOutput: P(flood), alert level', G)
arrow(2.3, 4.6, 2.65, 4.6); arrow(5.05, 4.6, 5.4, 4.6); arrow(8.0, 4.6, 8.35, 4.6)
arrow(10.1, 3.5, 1.9, 2.8)
arrow(3.7, 1.85, 4.0, 1.85); arrow(7.9, 1.85, 8.2, 1.85)
ax.set_title('Flood early-warning ML pipeline (SDG 13 Climate Action / SDG 11.5 disaster losses)', fontsize=12, weight='bold')
plt.tight_layout(); plt.savefig(f'{FIG}/architecture.png'); plt.close()
print('ok', len(cols))
