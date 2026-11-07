import os
import json
import random
import gc
import hashlib
import time
import zipfile
import shutil
import importlib.metadata
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, classification_report, confusion_matrix)

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric


import argparse
import yaml

def load_config(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

_parser = argparse.ArgumentParser(description='Train the DL models using a YAML configuration.')
_parser.add_argument('--config', default='configs/best.yaml')
_args = _parser.parse_args()
CONFIG_PATH = Path(_args.config).resolve()
CONFIG = load_config(CONFIG_PATH)
PROJECT_ROOT = CONFIG_PATH.parent.parent

data_cfg = CONFIG['data']
exp_cfg = CONFIG['experiment']
SPLIT_CONFIG = exp_cfg['split']
compute_cfg = CONFIG.get('compute', {})
output_cfg = CONFIG.get('outputs', {})
dl_cfg = CONFIG['deep_learning']

DATA_PATH = Path(data_cfg['path'])
if not DATA_PATH.is_absolute():
    DATA_PATH = PROJECT_ROOT / DATA_PATH
CSV_MEMBER = data_cfg.get('csv_member')
COMBINE_EXISTING_SPLITS = bool(data_cfg.get('combine_existing_splits', False))
ML_RESULTS_DIR = None
SEEDS = list(exp_cfg['seeds'])
TARGET = data_cfg.get('target', 'Severity')
REQUIRED_SEVERITY_LEVELS = list(data_cfg.get('required_severity_levels', [1, 2, 3, 4]))
MODEL_NAMES = list(dl_cfg['models'])
METRICS = ['Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_F1']
USE_CLASS_WEIGHT = bool(exp_cfg.get('use_class_weight', True))
CLASS_WEIGHT_CAP = float(exp_cfg.get('class_weight_cap', 2.5))
THRESHOLD_BETA = float(exp_cfg.get('threshold_beta', 1.0))
MIN_CATEGORY_COUNT = int(dl_cfg['min_category_count'])
MAX_CATEGORIES_PER_COLUMN = int(dl_cfg['max_categories_per_column'])
MAX_EPOCHS = int(dl_cfg['max_epochs'])
PATIENCE = int(dl_cfg['early_stopping_patience'])
NUM_WORKERS = int(dl_cfg.get('num_workers', 0))
REQUIRE_GPU = bool(dl_cfg.get('require_gpu', True))
USE_TWO_GPUS = bool(dl_cfg.get('use_two_gpus', False))
SAVE_MODELS = bool(dl_cfg.get('save_models', True))
SAVE_SPLIT_INDICES = bool(dl_cfg.get('save_split_indices', True))
SAVE_TEST_PREDICTIONS = bool(dl_cfg.get('save_test_predictions', True))
SAVE_TEST_PROBABILITIES = bool(dl_cfg.get('save_test_probabilities', True))
SAVE_SHAP = bool(dl_cfg.get('save_shap', False))
SHAP_BACKGROUND_ROWS = int(dl_cfg.get('shap_background_rows', 50))
SHAP_EXPLAIN_ROWS = int(dl_cfg.get('shap_explain_rows', 100))
PREDICT_BATCH_SIZE = int(dl_cfg.get('inference_batch_size', 100000))
GRADIENT_CLIP_NORM = float(dl_cfg.get('gradient_clip_norm', 1.0))
TABNET_VERBOSE = int(dl_cfg.get('tabnet_verbose', 1))
MODEL_CONFIGS = dl_cfg['model_params']
if set(MODEL_NAMES) != set(MODEL_CONFIGS):
    raise ValueError('deep_learning.models and deep_learning.model_params must contain the same model names.')

DATETIME_COLS = ['Start_Time', 'End_Time', 'Weather_Timestamp']
BOOL_COLS = ['Amenity','Crossing','Give_Way','Junction','No_Exit','Railway','Station','Stop','Traffic_Signal']
SELECTED_FEATURES = ['Start_Lat', 'Start_Lng', 'Street', 'City', 'County', 'State', 'Zipcode', 'Timezone', 'Airport_Code', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Direction', 'Wind_Speed(mph)', 'Weather_Condition', 'Amenity', 'Crossing', 'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Station', 'Stop', 'Traffic_Signal', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Year', 'Month', 'Hour', 'DayOfWeek', 'Season', 'Hour_sin', 'Hour_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Month_sin', 'Month_cos', 'Weather_Timestamp_diff_minutes']
CATEGORICAL_FEATURES = ['Street', 'City', 'County', 'State', 'Zipcode', 'Timezone', 'Airport_Code', 'Wind_Direction', 'Weather_Condition', 'Sunrise_Sunset', 'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight', 'Season']
NUMERICAL_FEATURES = [c for c in SELECTED_FEATURES if c not in CATEGORICAL_FEATURES]
CSV_DTYPES = {c: 'string' for c in CATEGORICAL_FEATURES}
assert 'Distance(mi)' not in SELECTED_FEATURES
assert len(SELECTED_FEATURES) == 41
assert len(SEEDS) == 5 and len(set(SEEDS)) == 5
assert MIN_CATEGORY_COUNT >= 1 and MAX_CATEGORIES_PER_COLUMN >= 1

RUN_ID = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
output_root = output_cfg.get('root')
OUTPUT_ROOT = Path(output_root) if output_root else Path.cwd()
if not OUTPUT_ROOT.is_absolute():
    OUTPUT_ROOT = PROJECT_ROOT / OUTPUT_ROOT
OUT_DIR = OUTPUT_ROOT / f'dl_repeated_splits_{RUN_ID}'
OUT_DIR.mkdir(parents=True, exist_ok=False)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if REQUIRE_GPU and DEVICE.type != 'cuda':
    raise RuntimeError('A CUDA GPU is required by deep_learning.require_gpu. Set it to false only for a small CPU test.')
GPU_COUNT = torch.cuda.device_count()
if USE_TWO_GPUS and GPU_COUNT < 2:
    raise RuntimeError('deep_learning.use_two_gpus is true but fewer than two CUDA GPUs are available.')
print('Config:', CONFIG_PATH, '| Data:', DATA_PATH)
print('Output:', OUT_DIR, '| Device:', DEVICE, '| GPUs:', GPU_COUNT)
if DEVICE.type == 'cuda':
    for i in range(GPU_COUNT):
        print(i, torch.cuda.get_device_name(i))
VERSIONS = {p: importlib.metadata.version(p) for p in [
    'numpy', 'pandas', 'scikit-learn', 'torch', 'pytorch-tabnet']}
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
protocol = {
    'method': 'repeated stratified holdouts', 'seeds': SEEDS,
    'fractions': {'train': float(SPLIT_CONFIG['train']),
                  'validation': float(SPLIT_CONFIG['validation']),
                  'test': float(SPLIT_CONFIG['test'])},
    'source': source_identity, 'original_rows': original_rows, 'usable_rows': len(X),
    'class_map': class_map, 'selected_features': list(X.columns), 'models': MODEL_NAMES,
    'class_weight_cap': CLASS_WEIGHT_CAP, 'threshold_beta': THRESHOLD_BETA,
    'threshold_rule': 'independent OvR F_beta; highest probability passing class; training-majority fallback',
    'early_stopping_metric': 'validation macro-F1', 'max_epochs': MAX_EPOCHS,
    'patience': PATIENCE, 'device': str(DEVICE), 'use_two_gpus': USE_TWO_GPUS,
    'category_min_count': MIN_CATEGORY_COUNT, 'category_max_count': MAX_CATEGORIES_PER_COLUMN,
    'model_configs': MODEL_CONFIGS, 'std_ddof': 1,
}
(OUT_DIR / 'protocol.json').write_text(json.dumps(protocol, indent=2))
if ML_RESULTS_DIR is not None:
    reference = json.loads((Path(ML_RESULTS_DIR) / 'protocol.json').read_text())
    for key in ['source', 'original_rows', 'usable_rows', 'selected_features']:
        if reference[key] != protocol[key]:
            raise ValueError(f'ML and DL differ in {key}; use the same full input and feature pipeline.')

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_preprocessor(frame):
    numerical = [c for c in NUMERICAL_FEATURES if c in frame]
    categorical = [c for c in CATEGORICAL_FEATURES if c in frame]
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    if numerical:
        imputed = imputer.fit_transform(frame[numerical])
        scaler.fit(imputed)
    mappings = {}
    for c in categorical:
        values = frame[c].astype('string').fillna('Missing')
        counts = values[values != 'Missing'].value_counts()
        retained = counts[counts >= MIN_CATEGORY_COUNT].head(MAX_CATEGORIES_PER_COLUMN).index
        # Include rare training categories with ID 2; unseen validation/test values receive 0.
        mapping = {str(value): 2 for value in counts.index}
        mapping.update({str(value): i + 3 for i, value in enumerate(retained)})
        mapping['Missing'] = 1
        mappings[c] = mapping
    return {'numerical': numerical, 'categorical': categorical, 'mappings': mappings,
            'imputer': imputer, 'scaler': scaler, 'input_columns': list(frame.columns),
            'cat_sizes': [max(3, max(mappings[c].values()) + 1) for c in categorical]}


def transform_dl(frame, bundle):
    if bundle['numerical']:
        numerical = bundle['scaler'].transform(bundle['imputer'].transform(frame[bundle['numerical']]))
        numerical = np.asarray(numerical, dtype='float32')
    else:
        numerical = np.empty((len(frame), 0), dtype='float32')
    categorical = np.empty((len(frame), len(bundle['categorical'])), dtype='int32')
    for i, c in enumerate(bundle['categorical']):
        categorical[:, i] = frame[c].astype('string').fillna('Missing').map(
            bundle['mappings'][c]).fillna(0).to_numpy(dtype='int32')
    assert np.isfinite(numerical).all()
    # Embedding IDs stay unscaled and must fit each train-fitted vocabulary.
    for i, size in enumerate(bundle['cat_sizes']):
        if len(categorical) and (categorical[:, i].min() < 0 or categorical[:, i].max() >= size):
            raise ValueError(f'Out-of-range embedding ID for {bundle["categorical"][i]}.')
    return numerical, categorical


def get_class_weights(labels):
    weights = np.ones(n_classes)
    if USE_CLASS_WEIGHT:
        weights = compute_class_weight('balanced', classes=np.arange(n_classes), y=labels)
        weights = np.minimum(weights, weights.min() * CLASS_WEIGHT_CAP)
        weights /= weights[labels].mean()
    return weights.astype('float32')


class AccidentDataset(Dataset):
    def __init__(self, numerical, categorical, labels=None):
        self.numerical = numerical
        self.categorical = categorical
        self.labels = labels

    def __len__(self):
        return len(self.numerical)

    def __getitem__(self, i):
        values = (torch.from_numpy(self.numerical[i]),
                  torch.from_numpy(self.categorical[i]).long())
        return values if self.labels is None else (*values, int(self.labels[i]))


def make_loader(numerical, categorical, labels, batch_size, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(AccidentDataset(numerical, categorical, labels), batch_size=batch_size,
        shuffle=shuffle, num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda',
        generator=generator)

class EmbeddingMLP(nn.Module):
    def __init__(self, n_numerical, cat_sizes, config):
        super().__init__()
        dims = [min(config['max_embedding_dim'], max(2, int(np.ceil(np.sqrt(size))))) for size in cat_sizes]
        self.embeddings = nn.ModuleList([nn.Embedding(size, dim) for size, dim in zip(cat_sizes, dims)])
        width = n_numerical + sum(dims)
        layers = []
        for hidden in config['hidden_dims']:
            layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(config['dropout'])]
            width = hidden
        layers.append(nn.Linear(width, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, numerical, categorical):
        embedded = [layer(categorical[:, i]) for i, layer in enumerate(self.embeddings)]
        return self.network(torch.cat([numerical] + embedded, dim=1))


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def create_torch_model(name, bundle):
    if name != 'MLP_Embeddings':
        raise ValueError(f'Unsupported PyTorch model: {name}')
    config = MODEL_CONFIGS[name]
    model = EmbeddingMLP(len(bundle['numerical']), bundle['cat_sizes'], config).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    if USE_TWO_GPUS and GPU_COUNT >= 2:
        model = nn.DataParallel(model, device_ids=[0, 1])
    return model, optimizer


@torch.inference_mode()
def evaluate_torch(model, numerical, categorical, labels, batch_size):
    model.eval()
    probabilities = np.empty((len(labels), n_classes), dtype='float32')
    total_loss, offset = 0., 0
    for num, cat, target in make_loader(numerical, categorical, labels, batch_size):
        num, cat, target = num.to(DEVICE), cat.to(DEVICE), target.to(DEVICE)
        logits = model(num, cat)
        total_loss += nn.functional.cross_entropy(logits, target, reduction='sum').item()
        p = logits.softmax(dim=1).cpu().numpy()
        probabilities[offset:offset + len(p)] = p
        offset += len(p)
    return total_loss / len(labels), f1_score(labels, probabilities.argmax(1), average='macro', zero_division=0)


@torch.inference_mode()
def predict_torch(model, numerical, categorical, batch_size):
    model.eval()
    output = np.empty((len(numerical), n_classes), dtype='float32')
    offset = 0
    for num, cat in make_loader(numerical, categorical, None, batch_size):
        p = model(num.to(DEVICE), cat.to(DEVICE)).softmax(dim=1).cpu().numpy()
        output[offset:offset + len(p)] = p
        offset += len(p)
    return output


def train_torch(name, train_data, val_data, y_train, y_val, bundle, weights, seed, run_dir):
    model, optimizer = create_torch_model(name, bundle)
    batch_size = MODEL_CONFIGS[name]['batch_size']
    loader = make_loader(*train_data, y_train, batch_size, shuffle=True, seed=seed)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=DEVICE), reduction='sum')
    best_score, stale = -np.inf, 0
    checkpoint = run_dir / f'{name}_best.pt'
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        weighted_sum, weight_sum = 0., 0.
        for num, cat, target in loader:
            num, cat, target = num.to(DEVICE), cat.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(num, cat)
            numerator = criterion(logits, target)
            denominator = criterion.weight[target].sum()
            loss = numerator / denominator
            if not torch.isfinite(loss):
                raise RuntimeError(f'{name}: nonfinite training loss.')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            weighted_sum += numerator.item()
            weight_sum += denominator.item()
        val_loss, val_f1 = evaluate_torch(model, *val_data, y_val, batch_size)
        history.append({'Epoch': epoch, 'Train_Weighted_Loss': weighted_sum / weight_sum,
                        'Validation_LogLoss': val_loss, 'Validation_Macro_F1': val_f1})
        pd.DataFrame(history).to_csv(run_dir / f'{name}_training_history.csv', index=False)
        print(f'{name} epoch {epoch}: train loss {weighted_sum/weight_sum:.4f}, val macro-F1 {val_f1:.4f}', flush=True)
        if val_f1 > best_score + 1e-6:
            best_score, stale = val_f1, 0
            torch.save(unwrap(model).state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    unwrap(model).load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    if not SAVE_MODELS:
        checkpoint.unlink()
    return model, pd.DataFrame(history)


class MacroF1(Metric):
    def __init__(self):
        self._name = 'macro_f1'
        self._maximize = True

    def __call__(self, y_true, y_score):
        return f1_score(y_true, y_score.argmax(axis=1), average='macro', zero_division=0)


def combine_tabnet(data):
    return np.concatenate(data, axis=1).astype('float32')


def train_tabnet(train_data, val_data, y_train, y_val, bundle, weights, seed, run_dir):
    config = MODEL_CONFIGS['TabNet']
    train_x, val_x = combine_tabnet(train_data), combine_tabnet(val_data)
    categorical_positions = list(range(len(bundle['numerical']), train_x.shape[1]))
    model = TabNetClassifier(n_d=config['n_d'], n_a=config['n_a'], n_steps=config['n_steps'],
        gamma=config['gamma'], cat_idxs=categorical_positions, cat_dims=bundle['cat_sizes'],
        cat_emb_dim=config['cat_emb_dim'], lambda_sparse=config['lambda_sparse'],
        optimizer_fn=torch.optim.AdamW,
        optimizer_params={'lr': config['lr'], 'weight_decay': config['weight_decay']},
        seed=seed, device_name=str(DEVICE), verbose=TABNET_VERBOSE)
    loss = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=DEVICE))
    # weights=0 disables the library's weighted sampling. Weighting is in the loss only.
    batch_size = min(config['batch_size'], len(y_train))
    virtual_batch = min(config['virtual_batch_size'], batch_size)
    model.fit(train_x, y_train, eval_set=[(val_x, y_val)], eval_name=['validation'],
        eval_metric=['logloss', MacroF1], loss_fn=loss, weights=0,
        max_epochs=MAX_EPOCHS, patience=PATIENCE, batch_size=batch_size,
        virtual_batch_size=virtual_batch, num_workers=NUM_WORKERS,
        drop_last=len(y_train) % batch_size == 1)
    history = pd.DataFrame({
        'Epoch': np.arange(1, len(model.history['loss']) + 1),
        'Train_Weighted_Loss': model.history['loss'],
        'Validation_LogLoss': model.history['validation_logloss'],
        'Validation_Macro_F1': model.history['validation_macro_f1'],
    })
    history.to_csv(run_dir / 'TabNet_training_history.csv', index=False)
    if SAVE_MODELS:
        model.save_model(str(run_dir / 'TabNet_best'))
    del train_x, val_x
    gc.collect()
    return model, history


def predict_dl(name, model, data):
    if name == 'TabNet':
        result = np.empty((len(data[0]), n_classes), dtype='float32')
        for start in range(0, len(result), PREDICT_BATCH_SIZE):
            end = min(start + PREDICT_BATCH_SIZE, len(result))
            result[start:end] = model.predict_proba(combine_tabnet((data[0][start:end], data[1][start:end])))
    else:
        result = predict_torch(model, *data, MODEL_CONFIGS[name]['batch_size'])
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), 1, atol=1e-4):
        raise RuntimeError('Invalid probabilities.')
    return result


def save_history_plot(history, name, run_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.Epoch, history.Train_Weighted_Loss, label='Training weighted loss')
    axes[0].plot(history.Epoch, history.Validation_LogLoss, label='Validation unweighted log loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss', title=name)
    axes[0].legend()
    axes[1].plot(history.Epoch, history.Validation_Macro_F1)
    axes[1].set(xlabel='Epoch', ylabel='Validation macro-F1', ylim=(0, 1))
    fig.tight_layout(); fig.savefig(run_dir / f'{name}_training_curves.png', dpi=180)
    plt.close(fig)

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

def save_shap_results(name, model, train_data, test_data, bundle, seed, run_dir, test_row_ids):
    if not SAVE_SHAP:
        return
    import shap
    rng = np.random.default_rng(seed)
    train_positions = rng.choice(len(train_data[0]), min(SHAP_BACKGROUND_ROWS, len(train_data[0])), replace=False)
    test_positions = rng.choice(len(test_data[0]), min(SHAP_EXPLAIN_ROWS, len(test_data[0])), replace=False)
    background = combine_tabnet((train_data[0][train_positions], train_data[1][train_positions]))
    samples = combine_tabnet((test_data[0][test_positions], test_data[1][test_positions]))
    n_num = len(bundle['numerical'])
    names = bundle['numerical'] + bundle['categorical']
    def probability_predictor(array):
        return predict_dl(name, model, (array[:, :n_num].astype('float32'), array[:, n_num:].astype('int32')))
    explainer = shap.Explainer(probability_predictor, shap.maskers.Independent(background, max_samples=len(background)),
                             algorithm='permutation', feature_names=names, seed=seed)
    explanation = explainer(samples, max_evals=2 * len(names) + 1)
    np.savez_compressed(run_dir / f'{name}_shap_values.npz', values=explanation.values,
        base_values=explanation.base_values, row_ids=test_row_ids[test_positions],
        feature_names=np.asarray(names, dtype=str), classes=np.asarray(classes))
    pd.DataFrame(np.abs(explanation.values).mean(axis=0), index=names,
                 columns=[f'Severity_{c}' for c in classes]).to_csv(run_dir / f'{name}_shap_mean_absolute.csv')

def split_for_seed(y, seed):
    """Stratified 64/16/20 split; reject any partition missing Severity 1–4."""
    y = np.asarray(y)
    expected = set(range(len(REQUIRED_SEVERITY_LEVELS)))
    if set(np.unique(y)) != expected:
        raise ValueError('Encoded labels must include all four Severity levels (1, 2, 3, 4).')
    positions = np.arange(len(y), dtype=np.int64)
    test_fraction = float(SPLIT_CONFIG['test'])
    train_fraction = float(SPLIT_CONFIG['train'])
    validation_fraction = float(SPLIT_CONFIG['validation'])
    if not np.isclose(train_fraction + validation_fraction + test_fraction, 1.0):
        raise ValueError('Configured train/validation/test fractions must sum to 1.')
    dev, test = train_test_split(positions, test_size=test_fraction, stratify=y, random_state=seed)
    val_fraction_of_dev = validation_fraction / (train_fraction + validation_fraction)
    train, val = train_test_split(dev, test_size=val_fraction_of_dev, stratify=y[dev], random_state=seed)
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
print('Preflight passed: every seed includes all four severity levels in every partition.')


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
for run_number, seed in enumerate(SEEDS, 1):
    run_dir = OUT_DIR / f'run_{run_number}_seed_{seed}'
    run_dir.mkdir()
    train_idx, val_idx, test_idx = split_for_seed(y, seed)
    counts = severity_counts_for_split(y, train_idx, val_idx, test_idx)
    print(counts)
    counts.to_csv(run_dir / 'split_severity_counts.csv')
    manifest = {'seed': seed, 'run': run_number, 'counts': {}, 'index_sha256': {}, 'class_counts': {}}
    for part, idx in [('train', train_idx), ('validation', val_idx), ('test', test_idx)]:
        manifest['counts'][part] = len(idx)
        manifest['index_sha256'][part] = hashlib.sha256(np.sort(row_ids[idx]).tobytes()).hexdigest()
        manifest['class_counts'][part] = {str(classes[c]): int((y[idx] == c).sum()) for c in range(n_classes)}
    if ML_RESULTS_DIR is not None:
        ml_manifests = list(Path(ML_RESULTS_DIR).glob(f'run_*_seed_{seed}/split_manifest.json'))
        if len(ml_manifests) != 1:
            raise ValueError(f'Expected one ML split manifest for seed {seed}.')
        reference = json.loads(ml_manifests[0].read_text())
        if manifest['index_sha256'] != reference['index_sha256']:
            raise ValueError(f'Seed {seed}: ML and DL split rows differ.')
    (run_dir / 'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    if SAVE_SPLIT_INDICES:
        np.savez_compressed(run_dir / 'split_row_ids.npz', train=row_ids[train_idx],
                            validation=row_ids[val_idx], test=row_ids[test_idx])
    bundle = fit_preprocessor(X.iloc[train_idx])
    joblib.dump(bundle, run_dir / 'preprocessing.pkl')
    pd.DataFrame({'Feature': bundle['categorical'], 'Category_ID_Count': bundle['cat_sizes']}).to_csv(
        run_dir / 'category_sizes.csv', index=False)
    train_data = transform_dl(X.iloc[train_idx], bundle)
    val_data = transform_dl(X.iloc[val_idx], bundle)
    test_data = transform_dl(X.iloc[test_idx], bundle)
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]
    weights = get_class_weights(y_train)
    (run_dir / 'class_weights.json').write_text(json.dumps(dict(zip(map(str, classes), weights.tolist())), indent=2))
    majority = int(np.bincount(y_train).argmax())
    for name in MODEL_NAMES:
        seed_everything(seed)
        print(f'Run {run_number}/{len(SEEDS)} | seed {seed} | {name}', flush=True)
        (run_dir / 'run_status.json').write_text(json.dumps({'status': 'training', 'model': name}))
        (run_dir / f'{name}_parameters.json').write_text(json.dumps(MODEL_CONFIGS[name], indent=2))
        started = time.perf_counter()
        try:
            if name == 'TabNet':
                model, history = train_tabnet(train_data, val_data, y_train, y_val, bundle, weights, seed, run_dir)
            else:
                model, history = train_torch(name, train_data, val_data, y_train, y_val, bundle, weights, seed, run_dir)
            fit_seconds = time.perf_counter() - started
            save_history_plot(history, name, run_dir)
            thresholds = tune_thresholds(y_val, predict_dl(name, model, val_data))
            (run_dir / f'{name}_thresholds.json').write_text(json.dumps({
                'thresholds': dict(zip(map(str, classes), thresholds.tolist())),
                'fallback_severity': classes[majority], 'beta': THRESHOLD_BETA,
                'objective': 'independent one-vs-rest F_beta',
                'rule': 'highest original probability among passing classes; otherwise training majority'}, indent=2))
            probabilities = predict_dl(name, model, test_data)
            before = probabilities.argmax(1)
            after = predict_with_thresholds(probabilities, thresholds, majority)
            scores = {}
            for stage, prediction in [('Before', before), ('After', after)]:
                scores[stage] = calc_metrics(y_test, prediction)
                metric_rows.append({'Run': run_number, 'Seed': seed, 'Model': name, 'Stage': stage,
                                    'Fit_seconds': fit_seconds, **scores[stage]})
                report_predictions(name, stage, y_test, prediction, run_dir, seed, run_number)
            delta_rows.append({'Run': run_number, 'Seed': seed, 'Model': name,
                **{m: scores['After'][m] - scores['Before'][m] for m in METRICS}})
            if SAVE_TEST_PREDICTIONS:
                pd.DataFrame({'Row_ID': row_ids[test_idx], 'y_true': np.asarray(classes)[y_test],
                    'Before': np.asarray(classes)[before], 'After': np.asarray(classes)[after]}).to_csv(
                        run_dir / f'{name}_test_predictions.csv', index=False)
            if SAVE_TEST_PROBABILITIES:
                np.savez_compressed(run_dir / f'{name}_test_probabilities.npz', row_ids=row_ids[test_idx],
                    classes=np.asarray(classes), y_true=np.asarray(classes)[y_test], probabilities=probabilities)
            save_shap_results(name, model, train_data, test_data, bundle, seed, run_dir, row_ids[test_idx])
            if name == 'TabNet':
                pd.DataFrame({'Feature': bundle['numerical'] + bundle['categorical'],
                              'Importance': model.feature_importances_}).sort_values('Importance', ascending=False).to_csv(
                                  run_dir / 'TabNet_feature_importance.csv', index=False)
            print(name, 'Before:', scores['Before'], 'After:', scores['After'], flush=True)
            flush_results()
        except Exception as exc:
            (run_dir / 'run_status.json').write_text(json.dumps({'status': 'failed', 'model': name, 'error': str(exc)}))
            raise
        del model, history, probabilities, before, after
        gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    (run_dir / 'run_status.json').write_text(json.dumps({'status': 'complete', 'seed': seed}))
    del train_data, val_data, test_data, bundle, weights
    gc.collect()

assert len(metric_rows) == 2 * len(SEEDS) * len(MODEL_NAMES)
assert len(delta_rows) == len(SEEDS) * len(MODEL_NAMES)
print('All requested DL runs completed.')

metrics_df=pd.DataFrame(metric_rows);deltas_df=pd.DataFrame(delta_rows);per_class_df=pd.DataFrame(per_class_rows)

def save_mean_std(frame,groupby,metrics,stem):
    summary=frame.groupby(groupby)[metrics].agg(['mean','std'])
    flat=summary.copy();flat.columns=[f'{m}_{stat}' for m,stat in flat.columns]
    flat.to_csv(OUT_DIR/f'{stem}_numeric.csv')
    formatted=pd.DataFrame(index=summary.index)
    for m in metrics:
        formatted[m]=[f'{a:.4f} ± {b:.4f}' for a,b in zip(summary[(m,'mean')],summary[(m,'std')])]
    formatted.to_csv(OUT_DIR/f'{stem}_scores.csv')
    print(formatted.to_string())
    return summary

print('Five repeated splits: test scores before threshold tuning')
before_summary=save_mean_std(metrics_df[metrics_df.Stage=='Before'],'Model',METRICS,'before_test')
print('Five repeated splits: test scores after threshold tuning')
after_summary=save_mean_std(metrics_df[metrics_df.Stage=='After'],'Model',METRICS,'after_test')
print('Five repeated splits: threshold impact (after − before)')
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
fig.suptitle('Five repeated splits: test mean ± sample SD before and after threshold tuning')
fig.tight_layout();fig.savefig(OUT_DIR/'before_after_scores.png',dpi=300);fig.savefig(OUT_DIR/'before_after_scores.pdf');plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(14,9))
for ax,m in zip(axes.flat,METRICS):
    table=delta_summary.reindex(MODEL_NAMES)
    ax.bar(x,table[(m,'mean')],yerr=table[(m,'std')].fillna(0),capsize=4,color='#228833')
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,MODEL_NAMES,rotation=25,ha='right')
    ax.set_title(m.replace('_',' '));ax.set_ylabel('After − before');ax.grid(axis='y',alpha=.2)
fig.suptitle('Five repeated splits: threshold tuning impact (mean ± sample SD)')
fig.tight_layout();fig.savefig(OUT_DIR/'threshold_impact_scores.png',dpi=300);plt.close(fig)
fig,axes=plt.subplots(1,len(MODEL_NAMES),figsize=(5*len(MODEL_NAMES),4.8),sharey=True)
for ax,name in zip(axes,MODEL_NAMES):
    for stage,color in [('Before','#4477AA'),('After','#EE6677')]:
        rows=metrics_df[(metrics_df.Model==name)&(metrics_df.Stage==stage)].sort_values('Run')
        ax.plot(rows.Run,rows.Macro_F1,'o-',label=stage,color=color)
    ax.set_title(name);ax.set_xticks(range(1,len(SEEDS)+1));ax.set_xlabel('Run');ax.set_ylim(0,1);ax.legend();ax.grid(alpha=.2)
axes[0].set_ylabel('Test Macro-F1');fig.tight_layout();fig.savefig(OUT_DIR/'macro_f1_by_run.png',dpi=300);plt.close(fig)
fig,axes=plt.subplots(1,len(MODEL_NAMES),figsize=(5*len(MODEL_NAMES),5),sharey=True)
for ax,name in zip(axes,MODEL_NAMES):
    for offset,stage,color in [(-width/2,'Before','#4477AA'),(width/2,'After','#EE6677')]:
        part=class_summary.loc[(name,stage)].reindex(classes)
        ax.bar(np.arange(len(classes))+offset,part[('F1','mean')],width,yerr=part[('F1','std')].fillna(0),capsize=3,label=stage,color=color)
    ax.set_title(name);ax.set_xticks(np.arange(len(classes)),classes);ax.set_xlabel('Severity');ax.set_ylim(0,1);ax.legend()
axes[0].set_ylabel('Per-class test F1');fig.suptitle('Five repeated splits: per-severity test F1 (mean ± sample SD)')
fig.tight_layout();fig.savefig(OUT_DIR/'per_severity_f1.png',dpi=300);plt.close(fig)
(OUT_DIR/'run_status.txt').write_text('SUCCESS: all requested seeds and three DL models completed.\n')
archive=shutil.make_archive(str(OUT_DIR),'zip',OUT_DIR)
print('Download results:',archive);print('Results archive:', archive)