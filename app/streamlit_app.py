"""
streamlit_app.py - Interface Web Premium para Well Log Lithology Prediction.

Dois modos de visualizacao:
  1. Previsor de Litologia (insere valores de logs e preve a rocha)
  2. Explorador de Dados (visualizacao analitica dos Well Logs)
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
                                          use_container_width=True)

    if submeter:
        log_values = {
            'GR': GR, 'SGR': SGR, 'RHOB': RHOB, 'NPHI': NPHI, 'PEF': PEF,
            'DRHO': DRHO, 'RDEP': RDEP, 'RMED': RMED, 'RSHA': RSHA,
            'RMIC': RMIC, 'RXO': RXO, 'DTC': DTC, 'DTS': DTS, 'CALI': CALI,
            'BS': BS, 'SP': SP, 'ROP': ROP, 'MUDWEIGHT': MUDWEIGHT, 'ROPA': ROPA, 'DCAL': DCAL
        }
        # Criar features derivadas
        gr_min, gr_max = 10, 200
        log_values['GR_NORM'] = (GR - gr_min) / (gr_max - gr_min + 1e-8)
        log_values['AI'] = RHOB * DTC
        log_values['NPHI_RHOB_RATIO'] = NPHI / (RHOB + 1e-8)
        log_values['RES_DIFF'] = RDEP - RSHA
        log_values['LOG_RDEP'] = np.log1p(max(RDEP, 0))
        log_values['DTS_DTC_RATIO'] = DTS / (DTC + 1e-8)
        log_values['DCAL_COMPUTED'] = CALI - BS

        try:
            from predict import predict_lithology
            result = predict_lithology(log_values)

            st.markdown("---")
            st.subheader("Resultado da Classificacao")

            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(render_litho_gauge(result['confidence'], result['lithology']),
                                use_container_width=True)
            with col2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                litho = result['lithology']
                conf = result['confidence']
                
                if result.get('physics_corrected', False):
                    st.warning("⚠️ **Correcao Fisica (PIML) Aplicada:** A predicao original estatistica foi anulada "
                               "porque violava regras petrofisicas basicas (ex: PEF ou GR inconsistente com a matriz).")
                               
                if conf > 0.7:
                    st.success(f"Classificacao de alta confianca: **{litho}** ({conf*100:.1f}%)")
                elif conf > 0.4:
                    st.warning(f"Classificacao moderada: **{litho}** ({conf*100:.1f}%). "
                               "Considere verificar os logs de entrada.")
                else:
                    st.error(f"Baixa confianca: **{litho}** ({conf*100:.1f}%). "
                             "Dados potencialmente inconsistentes.")

                # Top 5 probabilidades
                st.markdown("**Ranking de Probabilidades:**")
                for lith, prob in list(result['probabilities'].items())[:5]:
                    st.progress(prob, text=f"{lith}: {prob*100:.1f}%")
                    
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("🔍 Ver Todas as 27 Features Utilizadas (Motor PIML)"):
                st.markdown("O modelo XGBoost ingeriu os 20 logs de base fornecidos e calculou em tempo-real **7 features petrofísicas derivadas** para maximizar a separação geológica:")
                feat_df = pd.DataFrame([log_values]).T
                feat_df.columns = ["Valor"]
                # Formatar
                feat_df["Valor"] = feat_df["Valor"].apply(lambda x: f"{x:.4f}")
                st.dataframe(feat_df, use_container_width=True)

        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}. Corra primeiro o script src/train.py.")

elif modo == "Previsor em Lote (CSV)":
    st.title("Previsor em Lote (Importar CSV)")
    st.markdown("Carregue um ficheiro CSV contendo os logs petrofísicos **OU** insira/cole os dados manualmente na tabela abaixo.")
    
    uploaded_file = st.file_uploader("Opção 1: Upload de CSV", type=["csv"])
    
    # Colunas obrigatorias
    required_base_cols = ['GR', 'SGR', 'RHOB', 'NPHI', 'PEF', 'DRHO', 'RDEP', 'RMED', 'RSHA', 'RMIC', 'RXO', 'DTC', 'DTS', 'CALI', 'BS', 'SP', 'ROP', 'MUDWEIGHT', 'ROPA', 'DCAL']
    
    try:
        if uploaded_file is not None:
            df_initial = pd.read_csv(uploaded_file)
            st.success(f"Ficheiro carregado com {len(df_initial)} amostras. Pode editar os dados abaixo antes de prever.")
            # Garantir colunas
            for col in required_base_cols:
                if col not in df_initial.columns:
                    df_initial[col] = 0.0
        else:
            # Template vazio
            df_initial = pd.DataFrame(columns=required_base_cols)
            # Adicionar uma linha vazia para facilitar a edicao manual
            df_initial.loc[0] = 0.0
            
        st.markdown("### Opção 2: Edição Manual (Cole os seus dados aqui)")
        df_input = st.data_editor(df_initial, num_rows="dynamic", use_container_width=True)
        
        if not df_input.empty and st.button("Executar Predição em Lote (PIML Ativado)", type="primary"):
            with st.spinner("A processar e a aplicar Física aos dados..."):
                from predict import predict_lithology
                
                # Garantir que todas as colunas existem no df editado
                for col in required_base_cols:
                    if col not in df_input.columns:
                        df_input[col] = 0.0
                        
                results = []
                
                # Iterar linha a linha para passar pelas Hard Constraints
                for idx, row in df_input.iterrows():
                    log_values = row.to_dict()
                    
                    # Computar features derivadas
                    gr_min, gr_max = 10, 200
                    log_values['GR_NORM'] = (log_values['GR'] - gr_min) / (gr_max - gr_min + 1e-8)
                    log_values['AI'] = log_values['RHOB'] * log_values['DTC']
                    log_values['NPHI_RHOB_RATIO'] = log_values['NPHI'] / (log_values['RHOB'] + 1e-8)
                    log_values['RES_DIFF'] = log_values['RDEP'] - log_values['RSHA']
                    log_values['LOG_RDEP'] = np.log1p(max(log_values['RDEP'], 0))
                    log_values['DTS_DTC_RATIO'] = log_values['DTS'] / (log_values['DTC'] + 1e-8)
                    log_values['DCAL_COMPUTED'] = log_values['CALI'] - log_values['BS']
                    
                    res = predict_lithology(log_values)
                    
                    results.append({
                        'Predicted_Lithology': res['lithology'],
                        'Confidence': res['confidence'],
                        'Physics_Corrected': res.get('physics_corrected', False)
                    })
                    
                df_results = pd.DataFrame(results)
                df_final = pd.concat([df_input, df_results], axis=1)
                
                st.success("Predição Concluída!")
                
                c1, c2 = st.columns(2)
                c1.metric("Anomalias Petrofísicas Corrigidas", f"{df_results['Physics_Corrected'].sum()} amostras")
                c2.metric("Litologia Predominante", df_results['Predicted_Lithology'].mode()[0])
                
                st.dataframe(df_final[['Predicted_Lithology', 'Confidence', 'Physics_Corrected'] + list(df_input.columns)].head(100), use_container_width=True)
                    
                csv = df_final.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descarregar Resultados Completos (CSV)",
                    data=csv,
                    file_name='predicoes_litologia_batch.csv',
                    mime='text/csv',
                )
    except Exception as e:
        st.error(f"Erro ao processar o ficheiro: {e}")

elif modo == "Explorador de Dados":
    st.title("Explorador de Well Logs (FORCE 2020)")
    st.markdown("Visualizacao analitica dos dados de perfis de poco do Mar da Noruega.")

    try:
        df = pd.read_csv('data/processed/train_processed.csv')

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
        st.plotly_chart(fig, use_container_width=True)

        # Crossplot GR vs RHOB
        st.subheader("Crossplot: Gamma Ray vs Densidade")
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig2 = px.scatter(sample, x='GR', y='RHOB', color='LITHOLOGY',
                          color_discrete_map=LITHO_COLORS, opacity=0.5,
                          title="Separacao litologica no espaco GR-RHOB")
        fig2.update_layout(paper_bgcolor="#121212", plot_bgcolor="#121212",
                           font={'color': "#E0E0E0"})
        st.plotly_chart(fig2, use_container_width=True)

        st.info("Este crossplot classico (Gamma Ray vs Densidade) demonstra como diferentes "
                "litologias ocupam regioes distintas no espaco de features. "
                "O modelo XGBoost aprende estas fronteiras automaticamente a partir dos dados.")

    except FileNotFoundError:
        st.error("Dataset nao encontrado. Corra primeiro o script src/preprocess.py.")
