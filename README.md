# Well Log Lithology Prediction (FORCE 2020)

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Optimized-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow.svg)](https://scikit-learn.org/)

English Description Below | Descricao em Portugues Abaixo

---

## Descricao (Portugues)

Pipeline completa de Machine Learning para classificacao automatica de litologias a partir de dados de perfis de poco (Well Logs) do Mar da Noruega. O modelo recebe curvas geofisicas (Gamma Ray, Resistividade, Densidade, Neutrao, Sonico, etc.) e classifica automaticamente o tipo de rocha em 12 categorias litologicas.

### Principais Tecnologias
- **Python:** Linguagem principal da pipeline.
- **Pandas / NumPy:** Tratamento de dados e manipulacao matricial de logs de pocos.
- **Scikit-Learn:** Modelo baseline (Random Forest), pre-processamento e metricas de avaliacao (Classification Report).
- **XGBoost:** Modelo de Gradient Boosting optimizado para classificacao multi-classe.
- **Imbalanced-Learn (SMOTE):** Balanceamento sintetico para resolver classes extremamente raras.
- **Matplotlib / Plotly:** Visualizacao de dados geofisicos e apresentacao de resultados.
- **Streamlit:** Interface Web interactiva com dois modos de visualizacao.
- **Joblib:** Persistencia dos modelos treinados para deploy.

### Dataset
- **FORCE 2020 Machine Learning Lithology Prediction Competition** (Noruega)
- 1.17 milhoes de amostras de 98 pocos no Mar da Noruega
- 12 classes litologicas (Sandstone, Shale, Limestone, Chalk, Marl, Halite, Anhydrite, Tuff, Coal, Dolomite, Basement, Sandstone/Shale)
- Licenca: NOLD 2.0

### Resultados da Classificacao

| Modelo | Accuracy | F1-Score (Weighted) |
|--------|----------|---------------------|
| Random Forest (Baseline) | 94.19% | 94.08% |
| XGBoost + SMOTE | 88.02% | 88.27% |

O SMOTE sacrifica deliberadamente accuracy global para melhorar o Recall nas litologias raras.

**Metricas Detalhadas (XGBoost + SMOTE):**

| Litologia | Precision | Recall | F1-Score | Amostras (Teste) |
|-----------|-----------|--------|----------|------------------|
| Anhydrite | 0.94 | 0.83 | 0.89 | 217 |
| Basement | 1.00 | 0.95 | 0.97 | 20 |
| Chalk | 0.89 | 0.94 | 0.92 | 2,103 |
| Coal | 0.66 | 0.76 | 0.71 | 764 |
| Dolomite | 0.31 | 0.40 | 0.35 | 338 |
| Halite | 0.99 | 1.00 | 0.99 | 1,643 |
| Limestone | 0.76 | 0.74 | 0.75 | 11,264 |
| Marl | 0.71 | 0.84 | 0.77 | 6,666 |
| Sandstone | 0.86 | 0.87 | 0.87 | 33,787 |
| Sandstone/Shale | 0.71 | 0.81 | 0.76 | 30,091 |
| Shale | 0.96 | 0.91 | 0.93 | 144,161 |
| Tuff | 0.64 | 0.91 | 0.75 | 3,049 |

![Benchmark](reports/benchmark_comparison.png)
*Comparacao de Performance entre Random Forest e XGBoost + SMOTE.*

![Feature Importance](reports/feature_importance.png)
*Importancia relativa dos Well Logs na classificacao litologica.*

### Feature Engineering
O script de preprocessamento cria 7 features petrofisicas derivadas:
- **GR Normalizado:** Indice de argilosidade.
- **Impedancia Acustica (AI):** RHOB * DTC.
- **Ratio Neutrao-Densidade:** Separacao de litologias.
- **Diferenca de Resistividades:** Indicador de hidrocarbonetos.
- **Log Resistividade:** Melhoria da distribuicao.
- **Ratio DTS/DTC:** Indicador de tipo de fluido.
- **Delta Caliper:** Estabilidade do poco.

### Como Executar

```bash
git clone https://github.com/heltrakinho07/well-log-analysis.git
cd well-log-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
streamlit run app/streamlit_app.py
```

---

## Description (English)

End-to-end Machine Learning pipeline for automatic lithology classification from well log data in the Norwegian Sea. The model takes geophysical logging curves (Gamma Ray, Resistivity, Density, Neutron, Sonic, etc.) and automatically classifies the rock type across 12 lithological categories.

### Core Technologies
- **Python:** Main pipeline language.
- **Pandas / NumPy:** Data handling and matrix manipulation of well logs.
- **Scikit-Learn:** Baseline model (Random Forest), preprocessing and evaluation metrics (Classification Report).
- **XGBoost:** High-performance Gradient Boosting for multi-class classification.
- **Imbalanced-Learn (SMOTE):** Synthetic balancing to solve extreme class imbalances.
- **Matplotlib / Plotly:** Geophysical data visualization and results plotting.
- **Streamlit:** Interactive Web interface with two viewing modes.
- **Joblib:** Model persistence for deployment.

### Dataset
- **FORCE 2020 Machine Learning Lithology Prediction Competition** (Norway)
- 1.17 million samples from 98 wells in the Norwegian Sea
- 12 lithological classes (Sandstone, Shale, Limestone, Chalk, Marl, Halite, Anhydrite, Tuff, Coal, Dolomite, Basement, Sandstone/Shale)
- License: NOLD 2.0

### Classification Results

| Model | Accuracy | F1-Score (Weighted) |
|-------|----------|---------------------|
| Random Forest (Baseline) | 94.19% | 94.08% |
| XGBoost + SMOTE | 88.02% | 88.27% |

SMOTE deliberately trades overall accuracy for improved Recall on rare lithologies.

**Detailed Metrics (XGBoost + SMOTE):**

| Lithology | Precision | Recall | F1-Score | Test Samples |
|-----------|-----------|--------|----------|--------------|
| Anhydrite | 0.94 | 0.83 | 0.89 | 217 |
| Basement | 1.00 | 0.95 | 0.97 | 20 |
| Chalk | 0.89 | 0.94 | 0.92 | 2,103 |
| Coal | 0.66 | 0.76 | 0.71 | 764 |
| Dolomite | 0.31 | 0.40 | 0.35 | 338 |
| Halite | 0.99 | 1.00 | 0.99 | 1,643 |
| Limestone | 0.76 | 0.74 | 0.75 | 11,264 |
| Marl | 0.71 | 0.84 | 0.77 | 6,666 |
| Sandstone | 0.86 | 0.87 | 0.87 | 33,787 |
| Sandstone/Shale | 0.71 | 0.81 | 0.76 | 30,091 |
| Shale | 0.96 | 0.91 | 0.93 | 144,161 |
| Tuff | 0.64 | 0.91 | 0.75 | 3,049 |

### Feature Engineering
The preprocessing script creates 7 derived petrophysical features:
- **Normalized GR:** Shaliness index.
- **Acoustic Impedance (AI):** RHOB * DTC.
- **Neutron-Density Ratio:** Lithology separation.
- **Resistivity Difference:** Hydrocarbon indicator.
- **Log Resistivity:** Distribution improvement.
- **DTS/DTC Ratio:** Fluid type indicator.
- **Delta Caliper:** Wellbore stability.

### How to Run

```bash
git clone https://github.com/heltrakinho07/well-log-analysis.git
cd well-log-analysis
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/preprocess.py
python src/train.py
streamlit run app/streamlit_app.py
```
