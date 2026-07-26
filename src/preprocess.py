"""
preprocess.py - Limpeza e Feature Engineering dos dados de Well Logs (FORCE 2020).

Carrega o CSV bruto, limpa valores nulos, cria features petrofisicas
derivadas e exporta o dataset processado pronto para treino.
"""
import pandas as pd
import numpy as np
import os

# Mapeamento das classes litologicas (codigo numerico -> nome legivel)
LITHOLOGY_MAP = {
    30000: 'Sandstone',
    65030: 'Sandstone/Shale',
    65000: 'Shale',
    80000: 'Marl',
    74000: 'Dolomite',
    70000: 'Limestone',
    70032: 'Chalk',
    88000: 'Halite',
    86000: 'Anhydrite',
    99000: 'Tuff',
    90000: 'Coal',
    93000: 'Basement',
}

# Features de input (Well Logs geofisicos)
LOG_FEATURES = ['CALI', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'SGR',
                'NPHI', 'PEF', 'DTC', 'SP', 'BS', 'ROP', 'DTS',
                'DCAL', 'DRHO', 'MUDWEIGHT', 'RMIC', 'ROPA', 'RXO']

# Features nucleares (sempre disponiveis na maioria dos pocos)
CORE_FEATURES = ['GR', 'RHOB', 'NPHI', 'DTC', 'RDEP', 'PEF']


def load_raw_data(filepath='data/raw/train.csv'):
    """Carrega o CSV bruto do FORCE 2020."""
    print(f"A carregar dados brutos de: {filepath}")
    df = pd.read_csv(filepath, sep=';')
    print(f"  Dataset carregado: {df.shape[0]} amostras, {df.shape[1]} colunas.")
    print(f"  Pocos unicos: {df['WELL'].nunique()}")
    return df


def add_lithology_names(df):
    """Converte os codigos litologicos em nomes legiveis."""
    df['LITHOLOGY'] = df['FORCE_2020_LITHOFACIES_LITHOLOGY'].map(LITHOLOGY_MAP)
    unknown = df['LITHOLOGY'].isna().sum()
    if unknown > 0:
        print(f"  AVISO: {unknown} amostras com litologia desconhecida (removidas).")
        df = df.dropna(subset=['LITHOLOGY'])
    return df


def engineer_features(df):
    """Cria features petrofisicas derivadas (ratios e indices geologicos)."""
    print("  A criar features derivadas (ratios petrofisicos)...")

    # Ratio GR normalizado (indica argilosidade)
    gr_min = df['GR'].quantile(0.01)
    gr_max = df['GR'].quantile(0.99)
    df['GR_NORM'] = (df['GR'] - gr_min) / (gr_max - gr_min + 1e-8)
    df['GR_NORM'] = df['GR_NORM'].clip(0, 1)

    # Indice de Impedancia Acustica (RHOB * DTC)
    df['AI'] = df['RHOB'] * df['DTC']

    # Ratio Neutrao-Densidade (separacao de litologias)
    df['NPHI_RHOB_RATIO'] = df['NPHI'] / (df['RHOB'] + 1e-8)

    # Diferenca de Resistividades (indica presenca de hidrocarbonetos)
    df['RES_DIFF'] = df['RDEP'] - df['RSHA']

    # Log da Resistividade Profunda (melhor distribuicao)
    df['LOG_RDEP'] = np.log1p(df['RDEP'].clip(lower=0))

    # Ratio Sonico (DTS/DTC - indica tipo de fluido)
    df['DTS_DTC_RATIO'] = df['DTS'] / (df['DTC'] + 1e-8)

    # Diferenca Caliper (indica estabilidade do poco)
    df['DCAL_COMPUTED'] = df['CALI'] - df['BS']

    return df


def clean_and_process(df):
    """Pipeline completa de limpeza."""
    print("  A limpar dados...")

    # Seleccionar apenas as colunas relevantes
    feature_cols = LOG_FEATURES + ['WELL', 'DEPTH_MD', 'GROUP', 'FORMATION',
                                    'FORCE_2020_LITHOFACIES_LITHOLOGY']
    available = [c for c in feature_cols if c in df.columns]
    df = df[available].copy()

    # Converter litologia
    df = add_lithology_names(df)

    # Feature engineering
    df = engineer_features(df)

    # Preencher NaNs com a mediana de cada poco (estrategia robusta)
    log_cols = [c for c in LOG_FEATURES if c in df.columns]
    for col in log_cols:
        df[col] = df.groupby('WELL')[col].transform(
            lambda x: x.fillna(x.median())
        )
    # Restantes NaNs preenchidos com mediana global
    for col in log_cols:
        df[col] = df[col].fillna(df[col].median())

    derived_cols = ['GR_NORM', 'AI', 'NPHI_RHOB_RATIO', 'RES_DIFF',
                    'LOG_RDEP', 'DTS_DTC_RATIO', 'DCAL_COMPUTED']
    for col in derived_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df


def main():
    print("=" * 60)
    print("  WELL LOG ANALYSIS - Pre-processamento de Dados")
    print("=" * 60)

    df = load_raw_data()
    df = clean_and_process(df)

    # Estatisticas finais
    print(f"\n  Dataset processado: {df.shape[0]} amostras, {df.shape[1]} colunas.")
    print(f"\n  Distribuicao das classes litologicas:")
    print(df['LITHOLOGY'].value_counts().to_string())

    # Guardar
    out_path = 'data/processed/train_processed.csv'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n  Dataset processado guardado em: {out_path}")


if __name__ == '__main__':
    main()
