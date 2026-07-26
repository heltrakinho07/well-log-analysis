"""
train.py - Treino e avaliacao de modelos de classificacao litologica.

Treina um modelo baseline (Random Forest) e um modelo optimizado
(XGBoost + SMOTE) e compara os resultados.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Features de input
LOG_FEATURES = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR',
                'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS',
                'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']

DERIVED_FEATURES = ['GR_NORM', 'AI', 'NPHI_RHOB_RATIO', 'RES_DIFF',
                    'LOG_RDEP', 'DTS_DTC_RATIO', 'DCAL_COMPUTED']


def load_processed_data(filepath='data/processed/train_processed.csv'):
    """Carrega o dataset processado."""
    print(f"A carregar dados processados de: {filepath}")
    df = pd.read_csv(filepath)
    print(f"  {df.shape[0]} amostras carregadas.")
    return df


def prepare_features(df):
    """Prepara X e y para treino."""
    all_features = LOG_FEATURES + DERIVED_FEATURES
    available = [c for c in all_features if c in df.columns]
    X = df[available].copy()
    
    # Codificar o target
    le = LabelEncoder()
    y = le.fit_transform(df['LITHOLOGY'])
    
    print(f"  Features usadas: {len(available)}")
    print(f"  Classes: {list(le.classes_)}")
    
    return X, y, le, available


def train_baseline(X_train, y_train, X_test, y_test, le):
    """Treina o modelo baseline (Random Forest)."""
    print("\n" + "=" * 50)
    print("  MODELO BASELINE: Random Forest")
    print("=" * 50)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n  Accuracy: {acc*100:.2f}%")
    print(f"  F1-Score (weighted): {f1*100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    return rf, acc, f1


def train_xgboost_smote(X_train, y_train, X_test, y_test, le):
    """Treina o modelo optimizado (XGBoost + SMOTE)."""
    print("\n" + "=" * 50)
    print("  MODELO OPTIMIZADO: XGBoost + SMOTE")
    print("=" * 50)
    
    # Subamostragem estratificada para SMOTE (dataset completo e demasiado grande)
    print("  A criar subamostra estratificada para SMOTE (100K amostras)...")
    from sklearn.utils import resample
    sample_size = min(100000, len(X_train))
    X_sample, _, y_sample, _ = train_test_split(
        X_train, y_train, train_size=sample_size, random_state=42, stratify=y_train
    )
    
    print("  A aplicar SMOTE para balanceamento de classes...")
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = smote.fit_resample(X_sample, y_sample)
    print(f"  Amostras antes do SMOTE: {len(X_sample)}")
    print(f"  Amostras depois do SMOTE: {len(X_res)}")
    
    print("  A treinar XGBoost (300 arvores, profundidade 8)...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    xgb.fit(X_res, y_res)
    y_pred = xgb.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n  Accuracy: {acc*100:.2f}%")
    print(f"  F1-Score (weighted): {f1*100:.2f}%")
    print(f"\n{classification_report(y_test, y_pred, target_names=le.classes_)}")
    
    return xgb, acc, f1


def plot_comparison(rf_acc, rf_f1, xgb_acc, xgb_f1):
    """Gera grafico de comparacao entre modelos."""
    models = ['Random Forest\n(Baseline)', 'XGBoost + SMOTE\n(Optimizado)']
    acc_scores = [rf_acc * 100, xgb_acc * 100]
    f1_scores = [rf_f1 * 100, xgb_f1 * 100]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    colors = ['#4A90D9', '#F94C10']
    
    axes[0].bar(models, acc_scores, color=colors, edgecolor='black', linewidth=0.8)
    axes[0].set_title('Accuracy (%)', fontsize=14, fontweight='bold')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(acc_scores):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
    
    axes[1].bar(models, f1_scores, color=colors, edgecolor='black', linewidth=0.8)
    axes[1].set_title('F1-Score Weighted (%)', fontsize=14, fontweight='bold')
    axes[1].set_ylim(0, 100)
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold', fontsize=12)
    
    plt.suptitle('Benchmark: Well Log Lithology Classification', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/benchmark_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n  Grafico de comparacao guardado em reports/benchmark_comparison.png")


def plot_feature_importance(model, feature_names):
    """Gera grafico de importancia das features."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]  # Top 15
    
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(indices)), importances[indices], color='#F94C10', edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.invert_yaxis()
    ax.set_xlabel('Importancia Relativa', fontsize=12)
    ax.set_title('Feature Importance (XGBoost) - Top 15 Well Logs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Grafico de feature importance guardado em reports/feature_importance.png")


def main():
    print("=" * 60)
    print("  WELL LOG ANALYSIS - Treino de Modelos")
    print("=" * 60)
    
    df = load_processed_data()
    X, y, le, feature_names = prepare_features(df)
    
    # Divisao treino/teste (80/20) estratificada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Treino: {len(X_train)} | Teste: {len(X_test)}")
    
    # Treinar modelos
    rf_model, rf_acc, rf_f1 = train_baseline(X_train, y_train, X_test, y_test, le)
    xgb_model, xgb_acc, xgb_f1 = train_xgboost_smote(X_train, y_train, X_test, y_test, le)
    
    # Graficos
    plot_comparison(rf_acc, rf_f1, xgb_acc, xgb_f1)
    plot_feature_importance(xgb_model, feature_names)
    
    # Guardar o melhor modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(xgb_model, 'models/xgboost_lithology.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')
    print("\n  Modelo XGBoost guardado em models/xgboost_lithology.joblib")
    print("  Label Encoder guardado em models/label_encoder.joblib")
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETA!")
    print("=" * 60)


if __name__ == '__main__':
    main()
