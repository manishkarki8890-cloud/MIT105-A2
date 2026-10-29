# BaadhiSuchak – Next-day flood risk for Nepal river stations

Pipeline (run from `src/`, Python 3.10+, scikit-learn 1.9):

    python numbers_reader.py <dataset.numbers> ../data/nepal_raw.csv   # decode Apple Numbers -> CSV
    python preprocess.py                                               # cleaning, features, target
    python train.py                                                    # 3 models, CV, test metrics, figures, model
    python eda_figs.py                                                 # data-quality figures + architecture diagram

Note: the scripts use relative paths ../data and ../outputs; create an outputs/figs folder first.

Web app (Flask):

    cd app && pip install -r requirements.txt && python app.py         # http://127.0.0.1:5000
    # deploy: Render/Railway use the Procfile (gunicorn app:app)

After retraining, copy outputs/flood_model.joblib, outputs/results.json, data/thresholds.json and
data/nepal_features.csv into app/model/. The model was saved with scikit-learn 1.9.1 (pinned in
requirements.txt); load it with the same version.

Pages: / (dashboard), /station (replay), /predict (what-if), /about (model card), POST /api/predict (JSON).
Prototype for research/education - not an official DHM warning.
