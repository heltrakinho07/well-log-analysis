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

def apply_physics_constraints(prob_map: dict, log_values: dict):
    """
    Aplica Hard Constraints baseados em Petrofisica (PIML).
    Anula predicoes matematicas que sao geologicamente impossiveis.
    """
    pef = log_values.get('PEF', 0)
    gr = log_values.get('GR', 0)
    
    # Regra 1: Sandstone puro nao pode ter PEF > 3.0 (quartzo e 1.8)
    if pef > 3.0 and 'Sandstone' in prob_map:
        prob_map['Sandstone'] *= 0.1  # Penalizacao severa
        
    # Regra 2: Sandstone puro nao costuma ter GR alto (>85 API)
    if gr > 85 and 'Sandstone' in prob_map:
        prob_map['Sandstone'] *= 0.2
        
    # Regra 3: Shale costuma ter GR alto. Se o modelo prever Shale com GR < 40, penalizar.
    if gr < 40 and 'Shale' in prob_map:
        prob_map['Shale'] *= 0.1

    # Re-normalizar
    total_prob = sum(prob_map.values())
    if total_prob > 0:
        prob_map = {k: v / total_prob for k, v in prob_map.items()}
        
    return prob_map

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
    
    # Mapa de probabilidades inicial do XGBoost
    prob_map = {le.inverse_transform([i])[0]: float(p) for i, p in enumerate(proba)}
    
    # APLICAR PHYSICS-INFORMED ML (PIML)
    prob_map = apply_physics_constraints(prob_map, log_values)
    
    # Re-ordenar apos penalizacoes
    prob_map = dict(sorted(prob_map.items(), key=lambda x: x[1], reverse=True))
    
    # Nova predicao apos fisica
    best_litho = list(prob_map.keys())[0]
    best_conf = list(prob_map.values())[0]
    
    # Flag para saber se a fisica interveio (se a litologia inicial era diferente da litologia final)
    physics_corrected = (lithology != best_litho)
    
    return {
        "lithology": best_litho,
        "confidence": float(best_conf),
        "probabilities": prob_map,
        "physics_corrected": physics_corrected
    }
