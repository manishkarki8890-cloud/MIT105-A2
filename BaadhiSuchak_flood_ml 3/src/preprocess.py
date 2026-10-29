"""Step 1: data cleaning, feature engineering and flood-label creation.
Input : nepal_raw.csv (exported from the .numbers file by numbers_reader.py)
Output: nepal_clean.csv, nepal_features.csv, thresholds.json
"""
import json, sys
import numpy as np
import pandas as pd

RAW = sys.argv[1] if len(sys.argv) > 1 else '../data/nepal_raw.csv'
OUT = sys.argv[2] if len(sys.argv) > 2 else '../data'

# Upstream -> downstream station pairs (same river system)
UPSTREAM = {'Chatara': 'Bahrabise',      # Sun Koshi/Bhote Koshi -> Sapta Koshi
            'Devghat': 'Rasuwagadhi'}    # Trishuli/Bhote Koshi -> Narayani
TRAIN_END = '2024-12-31'                 # thresholds use training years only
FLOOD_Q = 0.95                           # station-specific high-flow percentile


def load_and_clean(path):
    df = pd.read_csv(path)
    log = {'rows_raw': len(df)}
    df.columns = [c.strip() for c in df.columns]                 # 'DHM Station ' -> 'DHM Station'
    df = df.rename(columns={'DHM Station': 'dhm_station'})
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
    log['duplicates'] = int(df.duplicated(['date', 'location']).sum())
    df = df.drop_duplicates(['date', 'location'])
    log['missing_cells'] = int(df.isna().sum().sum())

    # --- Fix 1: column rotation in all 2023 records -----------------------
    m = df['date'].dt.year == 2023
    src = df.loc[m, ['soil_moisture_0_100cm_m3m3', 'rain_mm', 'temperature_mean_c',
                     'dew_point_mean_c', 'precipitation_hours']].copy()
    df.loc[m, 'rain_mm'] = src['soil_moisture_0_100cm_m3m3'].values
    df.loc[m, 'precipitation_hours'] = src['rain_mm'].values
    df.loc[m, 'soil_moisture_0_100cm_m3m3'] = src['temperature_mean_c'].values
    df.loc[m, 'temperature_mean_c'] = src['dew_point_mean_c'].values
    df.loc[m, 'dew_point_mean_c'] = src['precipitation_hours'].values
    log['rows_column_shift_fixed'] = int(m.sum())

    # --- Fix 2: physically impossible values -> NaN, then interpolate -------
    rules = {'precipitation_hours': (0, 24), 'relative_humidity_mean_pct': (0, 100),
             'soil_moisture_0_100cm_m3m3': (0, 0.6), 'precipitation_mm': (0, 500),
             'rain_mm': (0, 500), 'river_discharge_m3s': (0, None),
             'wind_direction_dominant_deg': (0, 360)}
    inv = 0
    for c, (lo, hi) in rules.items():
        bad = (df[c] < lo) | ((df[c] > hi) if hi is not None else False)
        inv += int(bad.sum()); df.loc[bad, c] = np.nan
    log['invalid_values_set_nan'] = inv
    df = df.sort_values(['location', 'date'])
    num = df.select_dtypes('number').columns
    df[num] = df.groupby('location')[num].transform(lambda s: s.interpolate(limit_direction='both'))

    # --- Fix 3: wrong station elevations (gridded DEM artefacts) -----------
    # Kusum (West Rapti, inner Terai) and Chisapani (Karnali at the Siwalik gorge)
    # are lowland gauges; recorded 6367 m / 2215 m are impossible.
    elev_fix = {'Kusum': 200.0, 'Chisapani': 190.0}
    for loc, e in elev_fix.items():
        df.loc[df.location == loc, 'elevation_m'] = e
    log['elevation_corrections'] = elev_fix
    log['rows_clean'] = len(df)
    return df.reset_index(drop=True), log


def engineer(df):
    df = df.sort_values(['location', 'date']).copy()
    g = df.groupby('location')
    q = df['river_discharge_m3s']
    # rainfall memory (antecedent conditions)
    for k in (1, 2, 3):
        df[f'rain_lag{k}'] = g['precipitation_mm'].shift(k)
    for w in (3, 7, 14, 30):
        df[f'rain_sum_{w}d'] = g['precipitation_mm'].transform(lambda s: s.rolling(w, min_periods=1).sum())
    df['rain_hours_sum_3d'] = g['precipitation_hours'].transform(lambda s: s.rolling(3, min_periods=1).sum())
    df['soil_moist_change_3d'] = g['soil_moisture_0_100cm_m3m3'].diff(3)
    df['dewpoint_depression'] = df['temperature_mean_c'] - df['dew_point_mean_c']
    # discharge state, normalised by each station's training median (scale-free)
    med = df[df.date <= TRAIN_END].groupby('location')['river_discharge_m3s'].median().clip(lower=0.05)
    df['q_norm'] = q / df['location'].map(med)
    df['q_norm_lag1'] = df.groupby('location')['q_norm'].shift(1)
    df['q_norm_lag2'] = df.groupby('location')['q_norm'].shift(2)
    df['q_change_1d'] = df['q_norm'] - df['q_norm_lag1']
    df['q_norm_max_7d'] = df.groupby('location')['q_norm'].transform(lambda s: s.rolling(7, min_periods=1).max())
    # upstream signal
    df['upstream_q_norm'] = np.nan
    df['upstream_rain_3d'] = np.nan
    for down, up in UPSTREAM.items():
        u = df[df.location == up].set_index('date')
        idx = df.location == down
        df.loc[idx, 'upstream_q_norm'] = df.loc[idx, 'date'].map(u['q_norm']).values
        df.loc[idx, 'upstream_rain_3d'] = df.loc[idx, 'date'].map(u['rain_sum_3d']).values
    df['has_upstream'] = df['upstream_q_norm'].notna().astype(int)
    df[['upstream_q_norm', 'upstream_rain_3d']] = df[['upstream_q_norm', 'upstream_rain_3d']].fillna(0)
    # seasonality + circular wind direction
    doy = df['date'].dt.dayofyear
    df['doy_sin'], df['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    df['is_monsoon'] = df['date'].dt.month.between(6, 9).astype(int)
    wd = np.deg2rad(df['wind_direction_dominant_deg'])
    df['wind_dir_sin'], df['wind_dir_cos'] = np.sin(wd), np.cos(wd)

    # --- Target: will discharge TOMORROW exceed the station's P95? ---------
    thr = df[df.date <= TRAIN_END].groupby('location')['river_discharge_m3s'].quantile(FLOOD_Q)
    df['threshold_m3s'] = df['location'].map(thr)
    df['q_next'] = g['river_discharge_m3s'].shift(-1)
    df['flood_next_day'] = (df['q_next'] > df['threshold_m3s']).astype(int)
    df['flood_today'] = (q > df['threshold_m3s']).astype(int)
    # discharge relative to the station's own flood threshold (1.0 = at threshold)
    df['q_to_thr'] = q / df['threshold_m3s']
    df['q_to_thr_lag1'] = g['river_discharge_m3s'].shift(1) / df['threshold_m3s']
    df['q_to_thr_max_3d'] = df.groupby('location')['q_to_thr'].transform(lambda s: s.rolling(3, min_periods=1).max())
    df = df.dropna(subset=['q_next', 'rain_lag3', 'q_norm_lag2', 'soil_moist_change_3d'])
    return df.reset_index(drop=True), thr.to_dict(), med.to_dict()


if __name__ == '__main__':
    clean, log = load_and_clean(RAW)
    clean.to_csv(f'{OUT}/nepal_clean.csv', index=False)
    feat, thr, med = engineer(clean)
    feat.to_csv(f'{OUT}/nepal_features.csv', index=False)
    log['rows_model_ready'] = len(feat)
    log['positive_rate'] = round(float(feat.flood_next_day.mean()), 4)
    json.dump({'threshold_m3s': thr, 'median_m3s': med, 'upstream': UPSTREAM, 'log': log},
              open(f'{OUT}/thresholds.json', 'w'), indent=2)
    print(json.dumps(log, indent=2))
