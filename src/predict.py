"""
predict.py - Funcao de previsao reutilizavel para a interface Streamlit.
"""
import joblib
import pandas as pd
import numpy as np
import os

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

def load_model():
    """Carrega o modelo treinado e os artefactos."""
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_lithology.joblib'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.joblib'))
    features = joblib.load(os.path.join(MODEL_DIR, 'feature_names.joblib'))
    return model, le, features

def predict_lithology(log_values: dict):
    """
    Preve a litologia a partir de valores de Well Logs.
    
    Args:
        log_values: Dicionario {nome_do_log: valor}
    
    Returns:
        dict com litologia prevista e probabilidades
    """
    model, le, features = load_model()
    
    # Criar DataFrame com os valores fornecidos
    df = pd.DataFrame([log_values])
    
    # Garantir que todas as features existem (preencher com 0 as ausentes)
    for f in features:
        if f not in df.columns:
            df[f] = 0
    
    df = df[features]
    
    # Previsao
    pred = model.predict(df)[0]
    proba = model.predict_proba(df)[0]
    
    lithology = le.inverse_transform([pred])[0]
    
    # Mapa de probabilidades
    prob_map = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
    prob_map = dict(sorted(prob_map.items(), key=lambda x: x[1], reverse=True))
    
    return {
        "lithology": lithology,
        "confidence": float(max(proba)),
        "probabilities": prob_map
    }
