"""BaadhiSuchak - next-day flood-risk web app for 10 Nepal river stations.
Run:  pip install -r requirements.txt  &&  python app.py   ->  http://127.0.0.1:5000
"""
import json, os
import numpy as np, pandas as pd, joblib
from flask import Flask, render_template, request, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
bundle = joblib.load(os.path.join(HERE, 'model', 'flood_model.joblib'))
MODEL, FEATS, THR, NAME = bundle['model'], bundle['features'], bundle['threshold'], bundle['name']
# Replay of history uses the model trained on 2023-24 only, so dates from 2025 on are genuinely unseen.
EVAL_MODEL = bundle.get('model_eval', MODEL)
META = json.load(open(os.path.join(HERE, 'model', 'thresholds.json')))
HIST = pd.read_csv(os.path.join(HERE, 'model', 'nepal_features.csv'), parse_dates=['date'])
STATIONS = HIST.groupby('location').agg(lat=('latitude', 'first'), lon=('longitude', 'first'),
                                        river=('river', 'first'), basin=('basin', 'first'),
                                        elev=('elevation_m', 'first')).reset_index()
NUM = [f for f in FEATS if f != 'location']
DATES = (HIST.date.min().date().isoformat(), HIST.date.max().date().isoformat())
app = Flask(__name__)


def level(p):
    if p >= max(0.6, THR * 2.5): return 'Severe', 'sev'
    if p >= THR: return 'Warning', 'warn'
    if p >= THR / 2: return 'Watch', 'watch'
    return 'Normal', 'ok'


def predict_rows(X, replay=False):
    m = EVAL_MODEL if replay else MODEL
    return m.predict_proba(X[FEATS])[:, 1]


def build_manual(f):
    """Turn a short form (what a field officer knows) into the full 37-feature vector.
    Unknown features fall back to the station's median for that calendar month."""
    loc = f['location']; month = int(f.get('month', 7))
    base = HIST[(HIST.location == loc) & (HIST.date.dt.month == month)][NUM].median()
    row = base.copy()
    thr, med = META['threshold_m3s'][loc], max(META['median_m3s'][loc], 0.05)
    g = lambda k, d=None: float(f[k]) if f.get(k) not in (None, '') else d
    r0, r1, r2, r3 = g('rain_today', 0), g('rain_lag1', 0), g('rain_lag2', 0), g('rain_lag3', 0)
    q0, q1, q2 = g('q_today'), g('q_lag1'), g('q_lag2')
    q1 = q0 if q1 is None else q1; q2 = q1 if q2 is None else q2
    row['precipitation_mm'], row['rain_lag1'], row['rain_lag2'], row['rain_lag3'] = r0, r1, r2, r3
    row['rain_sum_3d'] = r0 + r1 + r2
    row['rain_sum_7d'] = g('rain_7d', max(row['rain_sum_3d'] + r3, row['rain_sum_7d']))
    row['rain_sum_14d'] = max(row['rain_sum_14d'], row['rain_sum_7d'])
    row['rain_sum_30d'] = g('rain_30d', max(row['rain_sum_30d'], row['rain_sum_14d']))
    row['precipitation_hours'] = g('rain_hours', row['precipitation_hours'])
    row['soil_moisture_0_100cm_m3m3'] = g('soil', row['soil_moisture_0_100cm_m3m3'])
    row['relative_humidity_mean_pct'] = g('rh', row['relative_humidity_mean_pct'])
    row['temperature_mean_c'] = g('temp', row['temperature_mean_c'])
    row['q_norm'], row['q_norm_lag1'], row['q_norm_lag2'] = q0 / med, q1 / med, q2 / med
    row['q_change_1d'] = row['q_norm'] - row['q_norm_lag1']
    row['q_norm_max_7d'] = max(row['q_norm'], row['q_norm_lag1'], row['q_norm_lag2'])
    row['q_to_thr'], row['q_to_thr_lag1'] = q0 / thr, q1 / thr
    row['q_to_thr_max_3d'] = max(q0, q1, q2) / thr
    row['flood_today'] = int(q0 > thr)
    doy = pd.Timestamp(2025, month, 15).dayofyear
    row['doy_sin'], row['doy_cos'] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    row['is_monsoon'] = int(6 <= month <= 9)
    X = pd.DataFrame([row]); X['location'] = loc
    return X, thr


def drivers(X, loc):
    """Simple local explanation: compare key inputs with the station's typical values."""
    typ = HIST[HIST.location == loc][NUM].median()
    out = []
    for k, label, unit in [('q_to_thr', 'Discharge vs flood threshold', '×'), ('rain_sum_3d', '3-day rainfall', 'mm'),
                           ('rain_sum_30d', '30-day rainfall', 'mm'), ('soil_moisture_0_100cm_m3m3', 'Soil moisture', 'm³/m³'),
                           ('q_change_1d', 'Discharge change (1 day)', '× median')]:
        v, t = float(X[k].iloc[0]), float(typ[k])
        out.append({'label': label, 'value': round(v, 3 if abs(v) < 2 else 1), 'typical': round(t, 3 if abs(t) < 2 else 1),
                    'unit': unit, 'high': v > t * 1.5 if t > 0 else v > 0.05})
    return out


@app.route('/')
def index():
    date = request.args.get('date', '2025-10-04')
    snap = HIST[HIST.date == date]
    cards = []
    if len(snap):
        snap = snap.assign(p=predict_rows(snap, replay=True))
        for _, r in snap.sort_values('p', ascending=False).iterrows():
            lv, cls = level(r.p)
            cards.append({'loc': r.location, 'river': r.river, 'p': r.p, 'level': lv, 'cls': cls,
                          'lat': r.latitude, 'lon': r.longitude, 'q': r.river_discharge_m3s,
                          'thr': r.threshold_m3s, 'rain3': r.rain_sum_3d,
                          'actual': 'Flood' if r.flood_next_day else 'No flood'})
    return render_template('index.html', cards=cards, date=date, dates=DATES, thr=THR, name=NAME,
                           insample=date < '2025-01-01', page='dash')


@app.route('/station')
def station():
    loc = request.args.get('loc', 'Chameliya/Nayalbadi')
    start = request.args.get('start', '2025-06-01'); end = request.args.get('end', '2025-10-31')
    s = HIST[(HIST.location == loc) & (HIST.date >= start) & (HIST.date <= end)].copy()
    s['p'] = predict_rows(s, replay=True) if len(s) else []
    series = [{'d': d.strftime('%Y-%m-%d'), 'q': round(q, 2), 'p': round(p, 3), 'y': int(y)}
              for d, q, p, y in zip(s.date, s.q_next, s.p, s.flood_next_day)]
    hits = int(((s.p >= THR) & (s.flood_next_day == 1)).sum()) if len(s) else 0
    stats = {'floods': int(s.flood_next_day.sum()) if len(s) else 0, 'hits': hits,
             'alerts': int((s.p >= THR).sum()) if len(s) else 0}
    return render_template('station.html', loc=loc, stations=STATIONS.location.tolist(), series=series,
                           start=start, end=end, thr=THR, thr_q=META['threshold_m3s'][loc], stats=stats,
                           dates=DATES, insample=start < '2025-01-01', page='station')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    res = None; form = request.form if request.method == 'POST' else {}
    if request.method == 'POST':
        X, thrq = build_manual(form)
        p = float(predict_rows(X)[0]); lv, cls = level(p)
        res = {'p': p, 'level': lv, 'cls': cls, 'thr_q': thrq, 'drivers': drivers(X, form['location'])}
    return render_template('predict.html', stations=STATIONS.to_dict('records'), res=res, f=form,
                           thr=THR, meta=META, page='predict')


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API: {"location": "Kusum", "month": 8, "rain_today": 60, "rain_lag1": 40, "q_today": 500, ...}"""
    f = request.get_json(force=True)
    X, thrq = build_manual({k: str(v) for k, v in f.items()})
    p = float(predict_rows(X)[0]); lv, _ = level(p)
    return jsonify({'location': f['location'], 'flood_probability': round(p, 4), 'alert_level': lv,
                    'alert_threshold': THR, 'station_flood_threshold_m3s': round(thrq, 2), 'model': NAME})


@app.route('/about')
def about():
    res = json.load(open(os.path.join(HERE, 'model', 'results.json')))
    return render_template('about.html', res=res, page='about', name=NAME, thr=THR)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
