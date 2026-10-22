"""Vectorized inference using the exact preprocessing fitted on training wells."""
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / 'models'

@lru_cache(maxsize=1)
def load_model():
    path = MODEL_DIR / 'lithology_bundle.joblib'
    if not path.exists():
        raise FileNotFoundError('Modelo actualizado indisponível. Execute python src/train.py com os dados FORCE. Os artefactos antigos não possuem pré-processamento compatível.')
    bundle = joblib.load(path)
    if bundle.get('schema_version') != 2:
        raise ValueError('Versão de modelo incompatível. Volte a treinar.')
    return bundle

def predict_batch(frame):
    bundle = load_model()
    X = bundle['preprocessor'].transform(frame)
    probabilities = bundle['model'].predict_proba(X)
    classes = bundle['label_encoder'].inverse_transform(bundle['model'].classes_.astype(int))
    rows = []
    for position, proba in enumerate(probabilities):
        best = int(np.argmax(proba))
        warnings = []
        if X.iloc[position]['RHOB'] > 2.90:
            warnings.append('Densidade elevada: verificar unidades, qualidade do log e contexto geológico.')
        if frame.iloc[position].reindex(X.columns[:20]).isna().any():
            warnings.append('Logs ausentes preenchidos com medianas do treino.')
        rows.append({'lithology': str(classes[best]), 'confidence': float(proba[best]),
                     'probabilities': {str(k): float(v) for k, v in zip(classes, proba)},
                     'physics_corrected': False, 'warnings': warnings})
    return rows

def predict_lithology(log_values):
    return predict_batch(pd.DataFrame([log_values]))[0]
