"""
streamlit_app.py - Interface Web Premium para Well Log Lithology Prediction.

Modos de visualizacao:
  1. Previsor de Litologia (insere valores de logs e preve a rocha)
  2. Previsor em Lote (CSV)
  3. Explorador de Dados (visualizacao analitica dos Well Logs)
"""
import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Permitir importar modulos em src/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

st.set_page_config(page_title="Well Log Lithology Predictor", page_icon="🪨", layout="wide")

# CSS para Dark Mode
st.markdown("""
<style>
    div[data-testid="stMetricValue"] { color: #F94C10; }
</style>
""", unsafe_allow_html=True)

# Cores litologicas (paleta geologica padrao)
LITHO_COLORS = {
    'Sandstone': '#F4D03F',
    'Sandstone/Shale': '#D4AC0D',
    'Shale': '#707B7C',
    'Marl': '#85C1E9',
    'Dolomite': '#AF7AC5',
    'Limestone': '#5DADE2',
    'Chalk': '#FADBD8',
    'Halite': '#F5B7B1',
    'Anhydrite': '#D2B4DE',
    'Tuff': '#A3E4D7',
    'Coal': '#2C3E50',
    'Basement': '#E74C3C',
}

# Sidebar
with st.sidebar:
    st.title("Well Log Analysis")
    st.markdown("Classificacao litologica automatica usando Machine Learning e dados de perfis de poco (FORCE 2020).")
    st.divider()
    modo = st.radio(
        "Modulo de Visualizacao:",
        ["Previsor de Litologia", "Previsor em Lote (CSV)", "Explorador de Dados"]
    )
    st.divider()
    st.caption("Desenvolvido por Helder Traquinho")
    st.caption("[GitHub](https://github.com/heltrakinho07)")


def render_litho_gauge(confidence, lithology):
    """Gauge chart para a confianca da previsao."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        number={"suffix": "%", "font": {"color": "#E0E0E0"}},
        title={'text': f"Litologia: {lithology}", 'font': {'size': 22, "color": "#E0E0E0"}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': "white"},
            'bar': {'color': LITHO_COLORS.get(lithology, '#F94C10')},
            'bgcolor': "#1E1E1E",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': 'rgba(255, 0, 0, 0.1)'},
                {'range': [40, 70], 'color': 'rgba(255, 255, 0, 0.1)'},
                {'range': [70, 100], 'color': 'rgba(0, 255, 0, 0.1)'}],
        }
    ))
    fig.update_layout(paper_bgcolor="#1E1E1E", font={'color': "#E0E0E0"}, height=350,
                      margin=dict(l=20, r=20, t=50, b=20))
    return fig


if modo == "Previsor de Litologia":
    st.title("Motor de Classificacao Litologica")
    st.markdown("Insira os valores dos perfis geofisicos (Well Logs) de uma profundidade "
                "especifica e o modelo XGBoost preve automaticamente o tipo de rocha.")

    with st.form("well_log_form"):
        tab1, tab2, tab3 = st.tabs(["Raios e Porosidade", "Resistividade", "Sonico e Outros"])

        with tab1:
            st.markdown("#### Logs de Raios Gamma e Porosidade")
            c1, c2, c3 = st.columns(3)
            with c1:
                GR = st.number_input("GR (Gamma Ray) [API]", 0.0, 300.0, 75.0)
                SGR = st.number_input("SGR (Spectral GR) [API]", 0.0, 300.0, 60.0)
            with c2:
                RHOB = st.number_input("RHOB (Densidade) [g/cm3]", 1.0, 3.5, 2.4)
                NPHI = st.number_input("NPHI (Neutrao) [v/v]", -0.1, 1.0, 0.25)
            with c3:
                PEF = st.number_input("PEF (Photo-Electric) [b/e]", 0.0, 10.0, 3.5)
                DRHO = st.number_input("DRHO (Delta Density)", -1.0, 1.0, 0.0)

        with tab2:
            st.markdown("#### Logs de Resistividade")
            c4, c5, c6 = st.columns(3)
            with c4:
                RDEP = st.number_input("RDEP (Resistividade Profunda) [ohm.m]", 0.01, 5000.0, 5.0)
                RMED = st.number_input("RMED (Resistividade Media) [ohm.m]", 0.01, 5000.0, 4.0)
            with c5:
                RSHA = st.number_input("RSHA (Resistividade Rasa) [ohm.m]", 0.01, 5000.0, 3.0)
                RMIC = st.number_input("RMIC (Micro-Resistividade) [ohm.m]", 0.0, 5000.0, 0.0)
            with c6:
                RXO = st.number_input("RXO (Resistividade Lavada) [ohm.m]", 0.0, 5000.0, 0.0)

        with tab3:
            st.markdown("#### Logs Sonicos e Mecanicos")
            c7, c8, c9 = st.columns(3)
            with c7:
                DTC = st.number_input("DTC (Sonico Compressional) [us/ft]", 40.0, 200.0, 90.0)
                DTS = st.number_input("DTS (Sonico Shear) [us/ft]", 0.0, 400.0, 150.0)
            with c8:
                CALI = st.number_input("CALI (Caliper) [in]", 5.0, 25.0, 8.5)
                BS = st.number_input("BS (Bit Size) [in]", 5.0, 25.0, 8.5)
            with c9:
                SP = st.number_input("SP (Self Potential) [mV]", -200.0, 200.0, 0.0)
                ROP = st.number_input("ROP (Rate of Penetration) [m/h]", 0.0, 200.0, 20.0)
                MUDWEIGHT = st.number_input("Mud Weight [ppg]", 8.0, 20.0, 10.0)
                ROPA = st.number_input("ROPA (ROP Average)", 0.0, 200.0, 0.0)
                DCAL = st.number_input("DCAL (Diff Caliper) [in]", -5.0, 10.0, 0.0)

        submeter = st.form_submit_button("Classificar Litologia", type="primary",
                                          width="stretch")

    if submeter:
        log_values = {
            'GR': GR, 'SGR': SGR, 'RHOB': RHOB, 'NPHI': NPHI, 'PEF': PEF,
            'DRHO': DRHO, 'RDEP': RDEP, 'RMED': RMED, 'RSHA': RSHA,
            'RMIC': RMIC, 'RXO': RXO, 'DTC': DTC, 'DTS': DTS, 'CALI': CALI,
            'BS': BS, 'SP': SP, 'ROP': ROP, 'MUDWEIGHT': MUDWEIGHT, 'ROPA': ROPA, 'DCAL': DCAL
        }
        try:
            from predict import predict_lithology
            result = predict_lithology(log_values)

            st.markdown("---")
            st.subheader("Resultado da Classificacao")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(render_litho_gauge(result['confidence'], result['lithology']),
                                width="stretch")
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                litho = result['lithology']
                conf = result['confidence']
                
                st.info(f"Pontuação do modelo: {conf*100:.1f}%. Esta probabilidade não foi calibrada como certeza geológica.")
                for warning in result['warnings']:
                    st.warning(warning)

                # Top 5 probabilidades
                st.markdown("**Ranking de Probabilidades:**")
                for lith, prob in sorted(result['probabilities'].items(), key=lambda item: item[1], reverse=True)[:5]:
                    st.progress(prob, text=f"{lith}: {prob*100:.1f}%")
                    
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔍 Ver as 27 Features Utilizadas"):
                st.markdown("O modelo XGBoost ingeriu os 20 logs de base fornecidos e calculou em tempo-real **7 features petrofísicas derivadas** para maximizar a separação geológica:")
                from predict import load_model
                feat_df = load_model()['preprocessor'].transform(pd.DataFrame([log_values])).T
                feat_df.columns = ["Valor"]
                # Formatar
                feat_df["Valor"] = feat_df["Valor"].apply(lambda x: f"{x:.4f}")
                st.dataframe(feat_df, width="stretch")

        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}. Corra primeiro o script src/train.py.")

elif modo == "Previsor em Lote (CSV)":
    st.title("Previsor em Lote (Importar CSV)")
    st.markdown("Carregue um ficheiro CSV contendo os logs petrofísicos **OU** insira os valores dos perfis geofísicos (Well Logs) de uma profundidade específica (na tabela abaixo) e o modelo XGBoost prevê automaticamente o tipo de rocha em massa.")
    
    uploaded_file = st.file_uploader("Opção 1: Upload de CSV", type=["csv"])
    
    from features import LOG_FEATURES, CORE_FEATURES
    st.caption('Logs obrigatórios: ' + ', '.join(CORE_FEATURES) + '. Use células vazias para dados ausentes; nunca zero como substituto.')
    try:
        if uploaded_file is not None:
            df_initial = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df_initial = pd.DataFrame([{name: np.nan for name in LOG_FEATURES}])
        df_input = st.data_editor(df_initial, num_rows="dynamic", width="stretch")
        if st.button("Executar Predição em Lote", type="primary"):
            st.session_state.pop('batch_result', None)
            with st.spinner("A classificar as amostras..."):
                from predict import predict_batch
                results = predict_batch(df_input)
                df_results = pd.DataFrame([{
                    'Predicted_Lithology': r['lithology'], 'Confidence': r['confidence'],
                    'Warnings': '; '.join(r['warnings'])} for r in results])
                st.session_state['batch_result'] = pd.concat(
                    [df_input.reset_index(drop=True), df_results], axis=1)
        if 'batch_result' in st.session_state:
            df_final = st.session_state['batch_result']
            st.caption('Resultados da última execução. Volte a executar após editar os dados. As probabilidades não são calibradas.')
            st.dataframe(df_final.head(100), width="stretch")
            st.download_button('Descarregar Resultados Completos (CSV)',
                data=df_final.to_csv(index=False).encode('utf-8'),
                file_name='predicoes_litologia_batch.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Erro ao processar os dados: {e}")

elif modo == "Explorador de Dados":
    st.title("Explorador de Well Logs (FORCE 2020)")
    st.markdown("Visualizacao analitica dos dados de perfis de poco do Mar da Noruega.")

    try:
        explorer_file = st.file_uploader('Carregar dados FORCE (opcional)', type=['csv'], key='explorer_csv')
        if explorer_file is not None:
            df = pd.read_csv(explorer_file, sep=None, engine='python')
            if 'LITHOLOGY' not in df.columns:
                from preprocess import add_lithology_names
                df = add_lithology_names(df)
        else:
            df = pd.read_csv(Path(__file__).resolve().parent.parent / 'data/processed/train_processed.csv')
        required = {'WELL', 'LITHOLOGY', 'GR', 'RHOB'}
        if not required.issubset(df.columns) or df.empty:
            raise ValueError('O explorador requer WELL, LITHOLOGY (ou o código FORCE), GR e RHOB e pelo menos uma amostra.')

        st.markdown("---")
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Total de Amostras", f"{len(df):,}")
        c_m2.metric("Pocos Unicos", f"{df['WELL'].nunique()}")
        c_m3.metric("Classes Litologicas", f"{df['LITHOLOGY'].nunique()}")
        c_m4.metric("Features de Input", "27")

        st.markdown("<br>", unsafe_allow_html=True)

        # Distribuicao Litologica
        st.subheader("Distribuicao das Classes Litologicas")
        counts = df['LITHOLOGY'].value_counts().reset_index()
        counts.columns = ['Litologia', 'Contagem']
        fig = px.bar(counts, x='Litologia', y='Contagem',
                     color='Litologia', color_discrete_map=LITHO_COLORS,
                     title="Frequencia de cada tipo de rocha no dataset")
        fig.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212",
                          font={'color': "#E0E0E0"}, showlegend=False)
        st.plotly_chart(fig, width="stretch")

        # Crossplot GR vs RHOB
        st.subheader("Crossplot: Gamma Ray vs Densidade")
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig2 = px.scatter(sample, x='GR', y='RHOB', color='LITHOLOGY',
                          color_discrete_map=LITHO_COLORS, opacity=0.5,
                          title="Separacao litologica no espaco GR-RHOB")
        fig2.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212",
                           font={'color': "#E0E0E0"})
        st.plotly_chart(fig2, width="stretch")

        st.info("Este crossplot classico (Gamma Ray vs Densidade) demonstra como diferentes "
                "litologias ocupam regioes distintas no espaco de features. "
                "O modelo XGBoost aprende estas fronteiras automaticamente a partir dos dados.")

    except FileNotFoundError:
        st.info("Carregue um CSV FORCE acima ou execute python src/preprocess.py para preparar o explorador.")
    except (ValueError, KeyError) as e:
        st.error(f'Dados incompatíveis com o explorador: {e}')
