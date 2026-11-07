# Generated from the 5-model repeated-stratified-holdout notebook.
# Run from the repository root: python train_ml.py --config configs/best.yaml

import json
import os
import importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
                             precision_recall_curve,classification_report,confusion_matrix)
from sklearn.utils.class_weight import compute_class_weight

import gc, hashlib, time, zipfile, shutil, subprocess
from datetime import datetime, timezone
import matplotlib.pyplot as plt


import argparse
import yaml

def load_config():
    parser = argparse.ArgumentParser(description="Train the five ML models using a YAML experiment config.")
    parser.add_argument("--config", default="configs/best.yaml", help="Path to YAML configuration.")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("The YAML configuration must contain a mapping.")
    return cfg

CONFIG = load_config()
DATA_PATH = Path(CONFIG["data"]["path"])
CSV_MEMBER = CONFIG["data"].get("csv_member")
COMBINE_EXISTING_SPLITS = bool(CONFIG["data"].get("combine_existing_splits", False))
TARGET = CONFIG["data"].get("target", "Severity")
REQUIRED_SEVERITY_LEVELS = CONFIG["data"].get("required_severity_levels", [1, 2, 3, 4])

SEEDS = list(CONFIG["experiment"]["seeds"])
SPLIT = CONFIG["experiment"]["split"]
USE_CLASS_WEIGHT = bool(CONFIG["experiment"]["use_class_weight"])
CLASS_WEIGHT_CAP = float(CONFIG["experiment"]["class_weight_cap"])
THRESHOLD_BETA = float(CONFIG["experiment"]["threshold_beta"])
HIGH_CARDINALITY_CUTOFF = int(CONFIG["experiment"]["high_cardinality_cutoff"])
TARGET_ENCODING_FOLDS = int(CONFIG["experiment"]["target_encoding_folds"])
PREDICT_BATCH_SIZE = int(CONFIG["experiment"]["predict_batch_size"])
CPU_THREADS = CONFIG["experiment"].get("cpu_threads") or max(1, os.cpu_count() or 1)

LGB_DEVICE = CONFIG["compute"]["lightgbm_device"]
XGB_DEVICE = CONFIG["compute"]["xgboost_device"]
CAT_DEVICE = CONFIG["compute"]["catboost_device"]
CAT_DEVICES = str(CONFIG["compute"]["catboost_devices"])

SAVE_SHAP = bool(CONFIG["outputs"]["save_shap"])
SHAP_BACKGROUND_ROWS = int(CONFIG["outputs"]["shap_background_rows"])
SHAP_EXPLAIN_ROWS = int(CONFIG["outputs"]["shap_explain_rows"])
SAVE_MODELS = bool(CONFIG["outputs"]["save_models"])
SAVE_TEST_PREDICTIONS = bool(CONFIG["outputs"]["save_test_predictions"])
SAVE_TEST_PROBABILITIES = bool(CONFIG["outputs"]["save_test_probabilities"])
SAVE_SPLIT_INDICES = bool(CONFIG["outputs"]["save_split_indices"])

DATETIME_COLS = ['Start_Time','End_Time','Weather_Timestamp']
BOOL_COLS = ['Amenity','Crossing','Give_Way','Junction','No_Exit','Railway','Station','Stop','Traffic_Signal']
METRICS = ['Accuracy','Macro_Precision','Macro_Recall','Macro_F1']
MODEL_NAMES = ['LogisticRegression','RandomForest','XGBoost','LightGBM','CatBoost']
MODEL_PARAMS = CONFIG["models"]

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
OUTPUT_ROOT = Path(CONFIG["outputs"].get("root") or Path.cwd())
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUT_DIR = OUTPUT_ROOT / f'ml_repeated_splits_{RUN_ID}'
OUT_DIR.mkdir(parents=True, exist_ok=False)
if len(SEEDS) != 5 or len(set(SEEDS)) != 5:
    raise ValueError("This experiment expects exactly five distinct seeds.")
print('Output:', OUT_DIR)


SELECTED_FEATURES = ['Start_Lat', 'Start_Lng', 'Street', 'City', 'County', 'State', 'Zipcode', 'Timezone', 'Airport_Code', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Weather_Condition', 'Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Station', 'Stop', 'Traffic_Signal', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Year', 'Month', 'Hour', 'DayOfWeek', 'Season', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos', 'Weather_Timestamp_diff_minutes']
CATEGORICAL_FEATURES = ['Street', 'City', 'County', 'State', 'Zipcode', 'Timezone', 'Airport_Code', 'Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Season']
NUMERICAL_FEATURES = [c for c in SELECTED_FEATURES if c not in CATEGORICAL_FEATURES]
CSV_DTYPES = {c: "string" for c in CATEGORICAL_FEATURES}

assert "Distance(mi)" not in SELECTED_FEATURES
assert len(SELECTED_FEATURES) == 41

def check_compute():
    global LGB_DEVICE, CAT_DEVICES
    needs_gpu = XGB_DEVICE.startswith('cuda') or CAT_DEVICE == 'GPU'
    if needs_gpu:
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError('Select GPU in Kaggle, or set XGB_DEVICE="cpu", CAT_DEVICE="CPU", LGB_DEVICE="cpu".') from exc
        if result.returncode != 0:
            raise RuntimeError('GPU unavailable. Select a Kaggle GPU accelerator.')
        print(result.stdout.strip())
        if len(result.stdout.strip().splitlines()) == 1 and CAT_DEVICES == '0:1':
            CAT_DEVICES = '0'
            print('One GPU detected: CatBoost devices changed to 0.')
    rng = np.random.default_rng(SEEDS[0])
    probe_x = rng.normal(size=(120, 6)).astype('float32')
    probe_y = np.tile(np.arange(4), 30)
    if XGB_DEVICE.startswith('cuda'):
        probe = xgb.XGBClassifier(n_estimators=2, max_depth=2, tree_method='hist',
                                 device=XGB_DEVICE, objective='multi:softprob', num_class=4)
        probe.fit(probe_x, probe_y)
        actual = json.loads(probe.get_booster().save_config())['learner']['generic_param']['device']
        if not actual.startswith('cuda'):
            raise RuntimeError(f'XGBoost unexpectedly fell back to {actual}.')
    if CAT_DEVICE == 'GPU':
        CatBoostClassifier(iterations=2, depth=2, loss_function='MultiClass',
                           task_type='GPU', devices=CAT_DEVICES, verbose=False,
                           allow_writing_files=False).fit(probe_x, probe_y)
    if LGB_DEVICE in ('auto', 'gpu'):
        requested = LGB_DEVICE
        try:
            lgb.LGBMClassifier(n_estimators=2, min_child_samples=1, max_bin=63,
                              device_type='gpu', verbosity=-1).fit(probe_x, probe_y)
            LGB_DEVICE = 'gpu'
        except Exception as exc:
            if requested != 'auto':
                raise RuntimeError('Explicit LightGBM GPU request failed; use a GPU-enabled build or CPU.') from exc
            LGB_DEVICE = 'cpu'
            print(f'LightGBM GPU unavailable; using CPU. Reason: {exc}')
    print(f'LR/RF: CPU | XGBoost: {XGB_DEVICE} | LightGBM: {LGB_DEVICE} | CatBoost: {CAT_DEVICE} {CAT_DEVICES}')

check_compute()
VERSIONS = {p: importlib.metadata.version(p) for p in
            ['numpy', 'pandas', 'scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'pyyaml']}
(OUT_DIR / 'package_versions.json').write_text(json.dumps(VERSIONS, indent=2))
def engineer_features(df):
    """Validate the already engineered 41-predictor schema; exclude Distance(mi)."""
    required = [TARGET] + SELECTED_FEATURES
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Processed dataset is missing required columns: {missing}')
    df = df[required].copy()
    for c in CATEGORICAL_FEATURES:
        # Identifier values remain categories even when their CSV values look numeric.
        values = df[c].astype('string').str.strip()
        values = values.mask(values.eq(''), pd.NA)
        df[c] = values.fillna('Missing').astype(str)
    boolean_map = {'true': 1., 'false': 0., '1': 1., '0': 0.,
                   '1.0': 1., '0.0': 0., 'y': 1., 'n': 0., 'yes': 1., 'no': 0.}
    for c in BOOL_COLS:
        values = df[c].astype('string').str.strip().str.lower()
        values = values.mask(values.eq(''), pd.NA)
        converted = values.map(boolean_map)
        invalid = values.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f'{c}: unsupported Boolean values {values[invalid].unique()[:5].tolist()}')
        df[c] = converted.astype('float32')
    for c in NUMERICAL_FEATURES:
        if c in BOOL_COLS:
            continue
        values = df[c]
        if not pd.api.types.is_numeric_dtype(values):
            values = values.astype('string').str.strip()
            values = values.mask(values.eq(''), pd.NA)
        converted = pd.to_numeric(values, errors='coerce')
        invalid = values.notna() & converted.isna()
        if invalid.any():
            raise ValueError(f'{c}: nonnumeric values in a numerical feature.')
        df[c] = converted.replace([np.inf, -np.inf], np.nan).astype('float32')
    # Missing feature values stay missing for training-fitted imputation.
    # Only missing targets are excluded; original row indices stay intact.
    df = df.dropna(subset=[TARGET])
    assert list(df.columns) == required
    return df


def save_feature_schema(frame):
    rows = []
    for c in SELECTED_FEATURES:
        values = frame[c]
        categorical = c in CATEGORICAL_FEATURES
        missing = values.eq('Missing') if categorical else values.isna()
        rows.append({'Feature': c,
                     'Role': 'categorical' if categorical else 'numerical',
                     'Dtype': str(values.dtype), 'Missing_count': int(missing.sum()),
                     'Missing_fraction': float(missing.mean()),
                     'Minimum': None if categorical else values.min(),
                     'Maximum': None if categorical else values.max()})
    pd.DataFrame(rows).to_csv(OUT_DIR / 'feature_schema.csv', index=False)
    (OUT_DIR / 'feature_schema.json').write_text(json.dumps({
        'target': TARGET, 'predictor_count': len(SELECTED_FEATURES),
        'feature_order': SELECTED_FEATURES, 'categorical': CATEGORICAL_FEATURES,
        'numerical': NUMERICAL_FEATURES}, indent=2))
    print(f'Feature schema: {len(SELECTED_FEATURES)} predictors | '
          f'{len(CATEGORICAL_FEATURES)} categorical | {len(NUMERICAL_FEATURES)} numerical', flush=True)

def load_full_data(path):
    if not path.is_file(): raise FileNotFoundError(f'Correct DATA_PATH: {path}')
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            members=[x for x in z.namelist() if x.lower().endswith('.csv') and not x.startswith('__MACOSX/')]
            if CSV_MEMBER is not None:
                if CSV_MEMBER not in members: raise ValueError(f'CSV_MEMBER not found. Available: {members}')
                selected=[CSV_MEMBER]
            elif len(members)==1: selected=members
            elif COMBINE_EXISTING_SPLITS:
                train=[x for x in members if 'train' in Path(x).name.lower()]
                test=[x for x in members if 'test' in Path(x).name.lower()]
                if len(train)!=1 or len(test)!=1 or train[0]==test[0]:
                    raise ValueError(f'Expected one train and one test member: {members}')
                selected=[train[0],test[0]]
            else: raise ValueError(f'ZIP has multiple CSVs: {members}. Set CSV_MEMBER or enable COMBINE_EXISTING_SPLITS.')
            frames=[]; identity=[]
            for member in selected:
                with z.open(member) as f: frame=pd.read_csv(f,low_memory=False,dtype=CSV_DTYPES)
                if TARGET not in frame: raise ValueError(f'{member} needs {TARGET}.')
                if frames and set(frame.columns)!=set(frames[0].columns): raise ValueError('CSV feature schemas differ.')
                frames.append(frame)
                info=z.getinfo(member)
                identity.append({'member':member,'crc32':f'{info.CRC:08x}','bytes':info.file_size,'rows':len(frame)})
            return pd.concat(frames,ignore_index=True),identity
    if path.suffix.lower() not in ['.csv','.gz']: raise ValueError('Use CSV, gzip CSV, or a ZIP of CSV files.')
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return pd.read_csv(path,low_memory=False,dtype=CSV_DTYPES),{'sha256':h.hexdigest(),'file':path.name}

raw, source_identity=load_full_data(DATA_PATH)
raw=raw.reset_index(drop=True)
original_rows=len(raw)
df=engineer_features(raw)
save_feature_schema(df)
del raw;gc.collect()
# Feature engineering retains original indices; labels missing from Severity are excluded.
row_ids=df.index.to_numpy(dtype=np.int64)
labels=pd.to_numeric(df[TARGET],errors='raise')
if not np.isfinite(labels).all() or not labels.isin([1,2,3,4]).all():
    raise ValueError('Severity must contain only 1,2,3,4 after removing missing labels.')
classes=sorted(labels.astype(int).unique().tolist())
missing_levels=sorted(set(REQUIRED_SEVERITY_LEVELS)-set(classes))
if missing_levels:
    raise ValueError(f'Full dataset is missing Severity levels {missing_levels}. All four levels are required; splitting cannot create missing classes.')
class_map={int(c):i for i,c in enumerate(classes)}
y=labels.astype(int).map(class_map).to_numpy(dtype=np.int64)
X=df.drop(columns=[TARGET]);del df;gc.collect()
n_classes=len(classes)
if n_classes<2: raise ValueError('At least two classes are required.')
counts=pd.Series(y).value_counts()
if counts.min()<10: raise ValueError('Not enough rows in a class for stable stratified 64/16/20 splitting.')
print('Full pool:',len(X),'| excluded missing-label rows:',original_rows-len(X),'| classes:',class_map)
protocol={'method':'repeated stratified holdouts','seeds':SEEDS,'fractions':SPLIT,
 'source':source_identity,'original_rows':original_rows,'usable_rows':len(X),'class_map':class_map,
 'LGB_DEVICE':LGB_DEVICE,'XGB_DEVICE':XGB_DEVICE,'class_weight_cap':CLASS_WEIGHT_CAP,
 'threshold_beta':THRESHOLD_BETA,'models':MODEL_NAMES,'CAT_DEVICE':CAT_DEVICE,'CAT_DEVICES':CAT_DEVICES,
 'threshold_rule':'independent OvR F_beta; highest probability passing class; training-majority fallback',
 'selected_features':list(X.columns),'std_ddof':1,'combine_existing_splits':COMBINE_EXISTING_SPLITS}
(OUT_DIR/'protocol.json').write_text(json.dumps(protocol,indent=2))
def calc_metrics(y_true, y_pred):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Macro_Precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro_Recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'Macro_F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }


def tune_thresholds(y_val, probabilities):
    """Preserve the original independent one-vs-rest validation F_beta objective."""
    thresholds = []
    for c in range(n_classes):
        precision, recall, candidates = precision_recall_curve(y_val == c, probabilities[:, c])
        beta2 = THRESHOLD_BETA ** 2
        denominator = beta2 * precision[:-1] + recall[:-1]
        scores = np.divide((1 + beta2) * precision[:-1] * recall[:-1], denominator,
                           out=np.zeros_like(denominator), where=denominator > 0)
        thresholds.append(float(candidates[np.argmax(scores)]))
    return np.asarray(thresholds)


def predict_with_thresholds(probabilities, thresholds, fallback_class):
    passed = probabilities >= thresholds[None, :]
    predictions = np.where(passed, probabilities, -np.inf).argmax(axis=1)
    predictions[~passed.any(axis=1)] = fallback_class
    return predictions


def class_weights(y_train):
    if not USE_CLASS_WEIGHT:
        return None
    weights = compute_class_weight('balanced', classes=np.arange(n_classes), y=y_train)
    # Cap relative ratios, then normalize mean training sample weight to 1.
    weights = np.minimum(weights, weights.min() * CLASS_WEIGHT_CAP)
    sample_weights = weights[y_train]
    return (sample_weights / sample_weights.mean()).astype('float32')


def add_frequency_features(frame, maps):
    frame = frame.copy()
    for column, mapping in maps.items():
        frame[column + '_freq'] = frame[column].map(mapping).fillna(0).astype('float32')
    return frame.drop(columns=list(maps))


def fit_preprocessor(x_train, y_train, seed):
    numerical = [c for c in NUMERICAL_FEATURES if c in x_train]
    categorical = [c for c in CATEGORICAL_FEATURES if c in x_train]
    high = [c for c in categorical if x_train[c].nunique() > HIGH_CARDINALITY_CUTOFF]
    low = [c for c in categorical if c not in high]
    maps = {c: x_train[c].value_counts(normalize=True) for c in high}
    prepared = add_frequency_features(x_train, maps)
    numerical += [c + '_freq' for c in high]
    transformers = []
    if numerical:
        transformers.append(('num', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), numerical))
    if low:
        # fit_transform cross-fits target encoding to prevent each row encoding its own label.
        transformers.append(('cat', TargetEncoder(target_type='multiclass', smooth='auto',
                             cv=TARGET_ENCODING_FOLDS, random_state=seed), low))
    preprocessor = ColumnTransformer(transformers, verbose_feature_names_out=True)
    encoded = np.asarray(preprocessor.fit_transform(prepared, y_train), dtype='float32')
    return encoded, {'transformer': preprocessor, 'frequency_maps': maps,
                     'input_columns': list(x_train.columns), 'numerical': numerical, 'categorical': low}


def transform_ml(frame, bundle):
    """Use this function for inference: restore frequency features before transforming."""
    prepared = add_frequency_features(frame[bundle['input_columns']], bundle['frequency_maps'])
    return np.asarray(bundle['transformer'].transform(prepared), dtype='float32')


def fit_cat_preprocessor(x_train):
    categorical = [c for c in CATEGORICAL_FEATURES if c in x_train]
    numerical = [c for c in x_train if c not in categorical]
    medians = x_train[numerical].median().fillna(0)
    return {'categorical': categorical, 'numerical': numerical, 'medians': medians,
            'input_columns': list(x_train.columns)}


def transform_cat(frame, bundle):
    prepared = frame[bundle['input_columns']].copy()
    for c in bundle['categorical']:
        prepared[c] = prepared[c].fillna('Missing').astype(str)
    for c in bundle['numerical']:
        prepared[c] = prepared[c].fillna(bundle['medians'][c]).astype('float32')
    return prepared



def make_model(name, seed):
    params = dict(MODEL_PARAMS[name])
    if name == 'LogisticRegression':
        return LogisticRegression(**params, random_state=seed)
    if name == 'RandomForest':
        return RandomForestClassifier(**params, n_jobs=CPU_THREADS, random_state=seed)
    if name == 'XGBoost':
        return xgb.XGBClassifier(**params, device=XGB_DEVICE, num_class=n_classes,
            n_jobs=CPU_THREADS, random_state=seed)
    if name == 'LightGBM':
        return lgb.LGBMClassifier(**params, device_type=LGB_DEVICE, gpu_device_id=0,
            verbosity=-1, n_jobs=CPU_THREADS, random_state=seed)
    if name == 'CatBoost':
        kwargs = dict(params, task_type=CAT_DEVICE, random_seed=seed, thread_count=CPU_THREADS,
                      verbose=False, allow_writing_files=False)
        if CAT_DEVICE == 'GPU':
            kwargs['devices'] = CAT_DEVICES
        return CatBoostClassifier(**kwargs)
    raise ValueError(name)



def fit_model(name, model, x_train, y_train, x_val, y_val, weights, cat_columns):
    if name == 'XGBoost':
        model.fit(x_train, y_train, sample_weight=weights,
                  eval_set=[(x_val, y_val)], verbose=False)
        history = model.evals_result()
    elif name == 'LightGBM':
        history = {}
        model.fit(x_train, y_train, sample_weight=weights, eval_set=[(x_val, y_val)],
                  eval_metric='multi_logloss',
                  callbacks=[lgb.record_evaluation(history)])
    elif name == 'CatBoost':
        model.fit(x_train, y_train, sample_weight=weights, cat_features=cat_columns,
                  eval_set=(x_val, y_val), verbose=False)
        history = model.get_evals_result()
    else:
        model.fit(x_train, y_train, sample_weight=weights)
        history = {}
    return history


def predict_probabilities(model, frame):
    output = np.empty((len(frame), n_classes), dtype='float32')
    for start in range(0, len(frame), PREDICT_BATCH_SIZE):
        stop = min(start + PREDICT_BATCH_SIZE, len(frame))
        batch = frame.iloc[start:stop] if isinstance(frame, pd.DataFrame) else frame[start:stop]
        output[start:stop] = model.predict_proba(batch)
    if not np.isfinite(output).all() or not np.allclose(output.sum(axis=1), 1, atol=1e-4):
        raise RuntimeError('Invalid predicted probabilities.')
    return output


def save_learning_history(history, name, run_dir):
    rows = [{'Dataset': dataset, 'Metric': metric, 'Iteration': i + 1, 'Value': value}
            for dataset, metrics in history.items() for metric, values in metrics.items()
            for i, value in enumerate(values)]
    if not rows:
        return  # LR/RF do not expose epoch-by-epoch validation histories.
    pd.DataFrame(rows).to_csv(run_dir / f'{name}_learning_history.csv', index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    for dataset, metrics in history.items():
        for metric, values in metrics.items():
            ax.plot(np.arange(1, len(values) + 1), values, label=f'{dataset}: {metric}')
    ax.set(xlabel='Boosting round', ylabel='Loss', title=name)
    ax.legend(); fig.tight_layout()
    fig.savefig(run_dir / f'{name}_learning_curve.png', dpi=160)
    plt.close(fig)

def save_shap_results(model, name, x_train, x_test, feature_names, seed, run_dir, cat_columns):
    if not SAVE_SHAP:
        return
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError('SAVE_SHAP requires shap: run %pip install shap before training.') from exc
    rng = np.random.default_rng(seed)
    background_idx = rng.choice(len(x_train), min(SHAP_BACKGROUND_ROWS, len(x_train)), replace=False)
    explain_idx = rng.choice(len(x_test), min(SHAP_EXPLAIN_ROWS, len(x_test)), replace=False)
    if isinstance(x_train, pd.DataFrame):
        background = x_train.iloc[background_idx].to_numpy()
        samples = x_test.iloc[explain_idx].to_numpy()
    else:
        background, samples = x_train[background_idx], x_test[explain_idx]
    # Categorical string arrays need an exact equality mask, rather than numpy.isclose.
    class ExactMasker(shap.maskers.Independent):
        def __init__(self, data, max_samples):
            super().__init__(data, max_samples=max_samples)
            self.supports_delta_masking = False

        def invariants(self, x):
            return self.data == x
    if name == 'CatBoost':
        def predict(array):
            frame = pd.DataFrame(array, columns=feature_names)
            for c in feature_names:
                frame[c] = frame[c].astype(str) if c in cat_columns else frame[c].astype('float32')
            return model.predict_proba(frame)
        masker = ExactMasker(background, max_samples=len(background))
    else:
        predict = model.predict_proba
        masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(predict, masker, algorithm='permutation',
                             feature_names=list(feature_names), seed=seed)
    explanation = explainer(samples, max_evals=2 * len(feature_names) + 1)
    np.savez_compressed(run_dir / f'{name}_shap_values.npz', values=explanation.values,
        base_values=explanation.base_values, test_positions=explain_idx,
        feature_names=np.asarray(feature_names, dtype=str), classes=np.asarray(classes))
    importance = np.abs(explanation.values).mean(axis=0)
    pd.DataFrame(importance, index=feature_names, columns=[f'Severity_{c}' for c in classes]).to_csv(
        run_dir / f'{name}_shap_mean_absolute.csv')
def split_for_seed(y, seed):
    """Stratified 64/16/20 split; reject any partition missing Severity 1–4."""
    y = np.asarray(y)
    expected = set(range(len(REQUIRED_SEVERITY_LEVELS)))
    if set(np.unique(y)) != expected:
        raise ValueError('Encoded labels must include all four Severity levels (1, 2, 3, 4).')
    positions = np.arange(len(y), dtype=np.int64)
    dev, test = train_test_split(positions, test_size=SPLIT['test'], stratify=y, random_state=seed)
    train, val = train_test_split(dev, test_size=SPLIT['validation'] / (SPLIT['train'] + SPLIT['validation']), stratify=y[dev], random_state=seed)
    combined = np.concatenate([train, val, test])
    if len(combined) != len(y) or len(np.unique(combined)) != len(y):
        raise RuntimeError(f'Seed {seed}: partitions overlap or do not cover the full dataset.')
    for name, idx in [('Train', train), ('Validation', val), ('Test', test)]:
        missing = expected - set(np.unique(y[idx]))
        if missing:
            levels = [REQUIRED_SEVERITY_LEVELS[c] for c in sorted(missing)]
            raise ValueError(f'Seed {seed}: {name} is missing Severity {levels}. '
                             'Add more records for the rare classes before running training.')
    return train, val, test

def severity_counts_for_split(y, train_idx, val_idx, test_idx):
    """Counts use original Severity labels, not the encoded 0–3 labels."""
    table = pd.DataFrame({
        name: np.bincount(y[idx], minlength=len(REQUIRED_SEVERITY_LEVELS))
        for name, idx in [('Train', train_idx), ('Validation', val_idx), ('Test', test_idx)]
    }, index=pd.Index(REQUIRED_SEVERITY_LEVELS, name='Severity'))
    if not (table.to_numpy() > 0).all():
        raise ValueError('Every partition must contain at least one record from each Severity level.')
    return table

# Validate every planned seed before any expensive model training begins.
for seed in SEEDS:
    split_for_seed(y, seed)
print('Preflight passed: every seed includes Severity 1, 2, 3, and 4 in all three partitions.')


def report_predictions(model_name, stage, y_true, prediction, run_dir, seed, run_number):
    matrix = confusion_matrix(y_true, prediction, labels=range(n_classes))
    pd.DataFrame(matrix, index=classes, columns=classes).rename_axis('Actual Severity').to_csv(
        run_dir / f'{model_name}_{stage}_confusion_matrix.csv')
    for normalized in (False, True):
        values = matrix / matrix.sum(axis=1, keepdims=True) if normalized else matrix
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(values, cmap='Blues', vmin=0, vmax=1 if normalized else None)
        for i in range(n_classes):
            for j in range(n_classes):
                ax.text(j, i, f'{values[i,j]:.2f}' if normalized else f'{values[i,j]:,}',
                        ha='center', va='center', fontsize=8,
                        color='white' if values[i,j] > values.max()/2 else 'black')
        ax.set(xticks=range(n_classes), yticks=range(n_classes), xticklabels=classes,
               yticklabels=classes, xlabel='Predicted Severity', ylabel='Actual Severity',
               title=f'{model_name} — {stage}')
        fig.colorbar(im, ax=ax); fig.tight_layout()
        suffix = '_normalized' if normalized else ''
        fig.savefig(run_dir / f'{model_name}_{stage}_confusion_matrix{suffix}.png', dpi=160)
        plt.close(fig)
    report = pd.DataFrame(classification_report(y_true, prediction, labels=range(n_classes),
        target_names=[str(c) for c in classes], output_dict=True, zero_division=0)).T
    report = report.drop(index='weighted avg', errors='ignore')
    # Binary one-vs-rest accuracy includes TP and TN for each encoded class.
    for encoded_class, label in enumerate(classes):
        report.loc[str(label), 'Accuracy'] = accuracy_score(
            np.asarray(y_true) == encoded_class, np.asarray(prediction) == encoded_class)
    report.to_csv(run_dir / f'{model_name}_{stage}_classification_report.csv')
    for label in classes:
        rec = report.loc[str(label)]
        per_class_rows.append({'Run': run_number, 'Seed': seed, 'Model': model_name,
            'Stage': stage, 'Severity': label, 'Accuracy': rec['Accuracy'],
            'Precision': rec['precision'],
            'Recall': rec['recall'], 'F1': rec['f1-score'], 'Support': int(rec['support'])})


def flush_results():
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / 'run_test_metrics.csv', index=False)
    pd.DataFrame(delta_rows).to_csv(OUT_DIR / 'run_threshold_deltas.csv', index=False)
    pd.DataFrame(per_class_rows).to_csv(OUT_DIR / 'run_per_class_metrics.csv', index=False)


metric_rows, delta_rows, per_class_rows = [], [], []
for run_number, current_seed in enumerate(SEEDS, 1):
    run_dir = OUT_DIR / f'run_{run_number}_seed_{current_seed}'
    run_dir.mkdir()
    train_idx, val_idx, test_idx = split_for_seed(y, current_seed)
    severity_table = severity_counts_for_split(y, train_idx, val_idx, test_idx)
    print(severity_table)
    severity_table.to_csv(run_dir / 'split_severity_counts.csv')
    manifest = {'seed': current_seed, 'run': run_number, 'counts': {}, 'index_sha256': {}, 'class_counts': {}}
    for part, idx in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        manifest['counts'][part] = len(idx)
        manifest['index_sha256'][part] = hashlib.sha256(np.sort(row_ids[idx]).tobytes()).hexdigest()
        manifest['class_counts'][part] = {str(classes[c]): int((y[idx] == c).sum()) for c in range(n_classes)}
    (run_dir / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    if SAVE_SPLIT_INDICES:
        np.savez_compressed(run_dir / 'split_row_ids.npz', train=row_ids[train_idx],
                            validation=row_ids[val_idx], test=row_ids[test_idx])
    ytr, yva, yte = y[train_idx], y[val_idx], y[test_idx]
    sw = class_weights(ytr)
    weight_table = {str(classes[c]): float(sw[ytr == c][0]) if sw is not None else 1.
                    for c in range(n_classes)}
    (run_dir / 'class_weights.json').write_text(json.dumps(weight_table, indent=2))
    majority = int(np.bincount(ytr).argmax())
    print(f'Run {run_number}/{len(SEEDS)}, seed {current_seed}', flush=True)
    # Prepare four shared encoded inputs once. CatBoost is last and uses native categories.
    Xtr_p, pre = fit_preprocessor(X.iloc[train_idx], ytr, current_seed)
    Xva_p = transform_ml(X.iloc[val_idx], pre)
    Xte_p = transform_ml(X.iloc[test_idx], pre)
    if SAVE_MODELS:
        joblib.dump(pre, run_dir / 'encoded_preprocess.pkl')
    for name in MODEL_NAMES:
        cat_columns = []
        if name == 'CatBoost':
            del Xtr_p, Xva_p, Xte_p, pre
            gc.collect()
            pre = fit_cat_preprocessor(X.iloc[train_idx])
            cat_columns = pre['categorical']
            Xtr_p = transform_cat(X.iloc[train_idx], pre)
            Xva_p = transform_cat(X.iloc[val_idx], pre)
            Xte_p = transform_cat(X.iloc[test_idx], pre)
            if SAVE_MODELS:
                joblib.dump(pre, run_dir / 'catboost_preprocess.pkl')
        model = make_model(name, current_seed)
        (run_dir / f'{name}_parameters.json').write_text(json.dumps(model.get_params(), indent=2, default=str))
        (run_dir / 'run_status.json').write_text(json.dumps({'status': 'running', 'model': name}))
        started = time.perf_counter()
        try:
            history = fit_model(name, model, Xtr_p, ytr, Xva_p, yva, sw, cat_columns)
        except Exception as exc:
            (run_dir / 'run_status.json').write_text(json.dumps({'status': 'failed', 'model': name, 'error': str(exc)}))
            raise
        fit_seconds = time.perf_counter() - started
        save_learning_history(history, name, run_dir)
        thresholds = tune_thresholds(yva, predict_probabilities(model, Xva_p))
        (run_dir / f'{name}_thresholds.json').write_text(json.dumps({
            'thresholds': dict(zip(map(str, classes), thresholds.tolist())),
            'fallback_severity': classes[majority], 'beta': THRESHOLD_BETA,
            'objective': 'independent one-vs-rest F_beta',
            'rule': 'highest original probability among passing classes; otherwise training majority',
        }, indent=2))
        test_proba = predict_probabilities(model, Xte_p)
        before = test_proba.argmax(axis=1)
        after = predict_with_thresholds(test_proba, thresholds, majority)
        scores = {}
        for stage, pred in [('Before', before), ('After', after)]:
            scores[stage] = calc_metrics(yte, pred)
            metric_rows.append({'Run': run_number, 'Seed': current_seed, 'Model': name,
                                'Stage': stage, 'Fit_seconds': fit_seconds, **scores[stage]})
            report_predictions(name, stage, yte, pred, run_dir, current_seed, run_number)
        delta_rows.append({'Run': run_number, 'Seed': current_seed, 'Model': name,
            **{m: scores['After'][m] - scores['Before'][m] for m in METRICS}})
        if SAVE_TEST_PREDICTIONS:
            pd.DataFrame({'Row_ID': row_ids[test_idx], 'y_true': np.asarray(classes)[yte],
                'Before': np.asarray(classes)[before], 'After': np.asarray(classes)[after]}).to_csv(
                    run_dir / f'{name}_test_predictions.csv', index=False)
        if SAVE_TEST_PROBABILITIES:
            np.savez_compressed(run_dir / f'{name}_test_probabilities.npz', row_ids=row_ids[test_idx],
                                classes=np.asarray(classes), y_true=np.asarray(classes)[yte], probabilities=test_proba)
        if name != 'LogisticRegression':
            names = list(X.columns) if name == 'CatBoost' else pre['transformer'].get_feature_names_out()
            pd.DataFrame({'Feature': names, 'Importance': model.feature_importances_}).sort_values(
                'Importance', ascending=False).to_csv(run_dir / f'{name}_feature_importance.csv', index=False)
        else:
            pd.DataFrame(model.coef_.T, index=pre['transformer'].get_feature_names_out(),
                         columns=[f'Severity_{c}' for c in classes]).to_csv(run_dir / f'{name}_coefficients.csv')
        feature_names = list(X.columns) if name == 'CatBoost' else pre['transformer'].get_feature_names_out()
        save_shap_results(model, name, Xtr_p, Xte_p, feature_names, current_seed, run_dir, cat_columns)
        if SAVE_MODELS:
            if name == 'CatBoost': model.save_model(str(run_dir / f'{name}.cbm'))
            elif name == 'XGBoost': model.save_model(str(run_dir / f'{name}.ubj'))
            else: joblib.dump(model, run_dir / f'{name}.pkl')
        print(name, 'Before:', scores['Before'], 'After:', scores['After'], flush=True)
        flush_results()
        del model, test_proba, before, after, history
        gc.collect()
    (run_dir / 'run_status.json').write_text(json.dumps({'status': 'complete', 'seed': current_seed}))
    del Xtr_p, Xva_p, Xte_p, pre, sw
    gc.collect()

assert len(metric_rows) == 2 * len(SEEDS) * len(MODEL_NAMES)
assert len(delta_rows) == len(SEEDS) * len(MODEL_NAMES)
print('All requested runs completed.')
metrics_df=pd.DataFrame(metric_rows);deltas_df=pd.DataFrame(delta_rows);per_class_df=pd.DataFrame(per_class_rows)

def save_mean_std(frame,groupby,metrics,stem):
    summary=frame.groupby(groupby)[metrics].agg(['mean','std'])
    flat=summary.copy();flat.columns=[f'{m}_{s}' for m,s in flat.columns]
    flat.to_csv(OUT_DIR/f'{stem}_numeric.csv')
    formatted=pd.DataFrame(index=summary.index)
    for m in metrics:
        formatted[m]=[f'{a:.4f} ± {b:.4f}' for a,b in zip(summary[(m,'mean')],summary[(m,'std')])]
    formatted.to_csv(OUT_DIR/f'{stem}_mean_std.csv')
    print(formatted)
    return summary

print('Before tuning: test mean ± std')
before_summary=save_mean_std(metrics_df[metrics_df.Stage=='Before'],'Model',METRICS,'before_test')
print('After tuning: test mean ± std')
after_summary=save_mean_std(metrics_df[metrics_df.Stage=='After'],'Model',METRICS,'after_test')
print('Paired impact (after − before): mean ± std')
delta_summary=save_mean_std(deltas_df,'Model',METRICS,'threshold_impact')
class_summary=save_mean_std(per_class_df,['Model','Stage','Severity'],['Accuracy','Precision','Recall','F1'],'per_class')
class_delta=per_class_df.pivot(index=['Run','Seed','Model','Severity'],columns='Stage',values=['Accuracy','Precision','Recall','F1'])
class_delta=pd.DataFrame({m:class_delta[(m,'After')]-class_delta[(m,'Before')] for m in ['Accuracy','Precision','Recall','F1']}).reset_index()
class_delta.to_csv(OUT_DIR/'run_per_class_deltas.csv',index=False)
class_delta_summary=save_mean_std(class_delta,['Model','Severity'],['Accuracy','Precision','Recall','F1'],'per_class_impact')

x=np.arange(len(MODEL_NAMES));width=.36
fig,axes=plt.subplots(2,2,figsize=(14,9))
for ax,m in zip(axes.flat,METRICS):
    for offset,summary,label,color in [(-width/2,before_summary,'Before','#4477AA'),(width/2,after_summary,'After','#EE6677')]:
        table=summary.reindex(MODEL_NAMES)
        ax.bar(x+offset,table[(m,'mean')],width,yerr=table[(m,'std')].fillna(0),capsize=4,label=label,color=color)
    ax.set_xticks(x,MODEL_NAMES,rotation=25,ha='right');ax.set_ylim(0,1);ax.set_ylabel('Test score');ax.set_title(m.replace('_',' '));ax.legend();ax.grid(axis='y',alpha=.2)
fig.suptitle(f'{len(SEEDS)} repeated splits: test mean ± sample std')
fig.tight_layout();fig.savefig(OUT_DIR/'before_after_mean_std.png',dpi=300);fig.savefig(OUT_DIR/'before_after_mean_std.pdf')
fig,axes=plt.subplots(2,2,figsize=(14,9))
for ax,m in zip(axes.flat,METRICS):
    table=delta_summary.reindex(MODEL_NAMES)
    ax.bar(x,table[(m,'mean')],yerr=table[(m,'std')].fillna(0),capsize=4,color='#228833')
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,MODEL_NAMES,rotation=25,ha='right')
    ax.set_title(m.replace('_',' '));ax.set_ylabel('After − before');ax.grid(axis='y',alpha=.2)
fig.suptitle('Threshold tuning: paired change mean ± sample std')
fig.tight_layout();fig.savefig(OUT_DIR/'threshold_impact_mean_std.png',dpi=300)
fig,axes=plt.subplots(1,len(MODEL_NAMES),figsize=(5*len(MODEL_NAMES),4.8),sharey=True)
for ax,name in zip(axes,MODEL_NAMES):
    for stage,color in [('Before','#4477AA'),('After','#EE6677')]:
        rows=metrics_df[(metrics_df.Model==name)&(metrics_df.Stage==stage)].sort_values('Run')
        ax.plot(rows.Run,rows.Macro_F1,'o-',label=stage,color=color)
    ax.set_title(name);ax.set_xticks(range(1,len(SEEDS)+1));ax.set_xlabel('Run');ax.set_ylim(0,1);ax.legend();ax.grid(alpha=.2)
axes[0].set_ylabel('Test Macro-F1');fig.tight_layout();fig.savefig(OUT_DIR/'macro_f1_by_run.png',dpi=300)
fig,axes=plt.subplots(1,len(MODEL_NAMES),figsize=(5*len(MODEL_NAMES),5),sharey=True)
for ax,name in zip(axes,MODEL_NAMES):
    for offset,stage,color in [(-width/2,'Before','#4477AA'),(width/2,'After','#EE6677')]:
        part=class_summary.loc[(name,stage)].reindex(classes)
        ax.bar(np.arange(len(classes))+offset,part[('F1','mean')],width,yerr=part[('F1','std')].fillna(0),capsize=3,label=stage,color=color)
    ax.set_title(name);ax.set_xticks(np.arange(len(classes)),classes);ax.set_xlabel('Severity');ax.set_ylim(0,1);ax.legend()
axes[0].set_ylabel('Per-class test F1');fig.suptitle(f'Per-severity F1: {len(SEEDS)}-run mean ± std')
fig.tight_layout();fig.savefig(OUT_DIR/'per_severity_f1_mean_std.png',dpi=300)
(OUT_DIR/'run_status.txt').write_text('SUCCESS: all requested seeds and five models completed.\n')
archive=shutil.make_archive(str(OUT_DIR),'zip',OUT_DIR)
print('Download results:',archive);print('Archive created:', archive)