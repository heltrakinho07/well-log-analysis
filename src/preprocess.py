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


def clean_and_process(df):
    """Map targets only. Fitted transformations belong to the training split."""
    return add_lithology_names(df.copy())


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
