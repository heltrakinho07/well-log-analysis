"""Train and evaluate on disjoint wells; preprocessing is fitted only on training."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from features import LogPreprocessor
from preprocess import load_raw_data, add_lithology_names

ROOT = Path(__file__).resolve().parent.parent

def split_wells(df):
    if 'WELL' not in df or df.WELL.isna().any() or df.WELL.nunique() < 2:
        raise ValueError('São necessários pelo menos dois poços identificados sem WELL vazio.')
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(df, groups=df.WELL))
    return train, test

def balanced_sample(X, y, limit=100000):
    if len(X) <= limit:
        return X, y
    classes = np.unique(y)
    if limit < len(classes):
        raise ValueError('O limite de subamostragem deve incluir todas as classes.')
    if pd.Series(y).value_counts().min() >= 2 and len(y)-limit >= len(classes):
        a, _, b, _ = train_test_split(X, y, train_size=limit, random_state=42, stratify=y)
        return a, b
    rng = np.random.default_rng(42)
    # Reserve one real example per class, including singleton classes.
    reserved = np.array([rng.choice(np.flatnonzero(y == label)) for label in classes])
    remaining = np.setdiff1d(np.arange(len(y)), reserved)
    selected = np.concatenate([reserved, rng.choice(remaining, limit-len(reserved), replace=False)])
    rng.shuffle(selected)
    return X.iloc[selected], y[selected]

def resample_training(X, y):
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    counts = pd.Series(y).value_counts()
    if len(counts) < 2:
        raise ValueError('O treino deve conter pelo menos duas classes.')
    if counts.min() < 2:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    return SMOTE(random_state=42, k_neighbors=min(3, int(counts.min())-1)).fit_resample(X, y)

def evaluate(model, X, y, encoder):
    predicted = model.predict(X)
    labels = np.arange(len(encoder.classes_))
    return {'accuracy': accuracy_score(y, predicted),
            'f1_weighted': f1_score(y, predicted, average='weighted', zero_division=0),
            'f1_macro': f1_score(y, predicted, labels=labels, average='macro', zero_division=0),
            'classification_report': classification_report(y, predicted, labels=labels,
                target_names=encoder.classes_, output_dict=True, zero_division=0)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/raw/train.csv')
    args = parser.parse_args()
    from xgboost import XGBClassifier
    print('A carregar e separar poços...', flush=True)
    df = add_lithology_names(load_raw_data(args.data))
    train, test = split_wells(df)
    preprocessor = LogPreprocessor().fit(df.iloc[train])
    X_train, X_test = preprocessor.transform(df.iloc[train]), preprocessor.transform(df.iloc[test])
    encoder = LabelEncoder().fit(df.iloc[train].LITHOLOGY)
    unknown = set(df.iloc[test].LITHOLOGY) - set(encoder.classes_)
    if unknown:
        raise ValueError(f'Classes presentes apenas no teste: {sorted(unknown)}. Reveja a cobertura de poços; não é possível avaliar essas classes com este treino.')
    y_train, y_test = encoder.transform(df.iloc[train].LITHOLOGY), encoder.transform(df.iloc[test].LITHOLOGY)
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print('A treinar Random Forest...', flush=True)
    rf.fit(X_train, y_train)
    sample_X, sample_y = balanced_sample(X_train, y_train)
    sample_X, sample_y = resample_training(sample_X, sample_y)
    xgb = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=.1, subsample=.8,
                        colsample_bytree=.8, random_state=42, n_jobs=-1, tree_method='hist')
    print('A treinar XGBoost com dados balanceados...', flush=True)
    xgb.fit(sample_X, sample_y)
    # Keep the holdout strictly for reporting; deployment choice is explicit, not test-selected.
    metrics = {name: evaluate(model, X_test, y_test, encoder) for name, model in [('random_forest', rf), ('xgboost_smote', xgb)]}
    metrics.update({'deployment_model': 'xgboost_smote', 'selection_policy': 'Predeclared; compare macro-F1 and per-class recall before changing.',
        'train_wells': sorted(df.iloc[train].WELL.unique().tolist()), 'test_wells': sorted(df.iloc[test].WELL.unique().tolist())})
    metrics['versions'] = {name: importlib.metadata.version(name) for name in ['scikit-learn', 'xgboost', 'imbalanced-learn', 'pandas', 'numpy']}
    with args.data.open('rb') as source:
        metrics['data_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    (ROOT/'models').mkdir(exist_ok=True)
    (ROOT/'reports').mkdir(exist_ok=True)
    joblib.dump({'schema_version': 2, 'preprocessor': preprocessor, 'model': xgb,
                 'label_encoder': encoder}, ROOT/'models/lithology_bundle.joblib', compress=3)
    (ROOT/'reports/metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    from reporting import write_figures
    write_figures(metrics, xgb, X_train.columns.tolist(), ROOT/'reports')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
