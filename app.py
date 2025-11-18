import streamlit as st
import os
import io
import zipfile
import plotly.graph_objects as go
from plotly.colors import qualitative
import scipy.io as sio
import numpy as np
import pandas as pd

# ===================== Configuração da Página =====================
st.set_page_config(layout="wide", page_title="Analisador de Harmônicas Trifásico")
st.title("Analisador Avançado de Qualidade de Energia (Trifásico)")

# ===================== Constantes =====================
F_FUNDAMENTAL = 60  # Hz
F_MAX_ANALISE = 2000 # Analisar THD até 2kHz
HARMONICOS_IMPARES = [h for h in range(3, int(F_MAX_ANALISE / F_FUNDAMENTAL) + 1, 2) if h <= 15]

PONTOS_BASE = ['800', 'T2F', '818_1', '818_2', '820', '822']
METRICAS_TYPES = ['Pico', 'THD', 'H3', 'H5', 'H7']
FASES = ['A', 'B', 'C']

# ===================== Funções Auxiliares =====================
@st.cache_data
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    if freq_array.size == 0: return 0
    target_freq = order * fund_freq
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]

@st.cache_data
def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    if freq_array.size == 0: return 0
    amp_h1 = get_harmonic_amplitude(freq_array, amp_array, 1, fund_freq)
    if amp_h1 == 0: return 0
    
    sum_squares = 0
    for order in range(2, int(max_freq/fund_freq)):
        amp_hn = get_harmonic_amplitude(freq_array, amp_array, order, fund_freq)
        sum_squares += amp_hn ** 2
        
    thd = (np.sqrt(sum_squares) / amp_h1) * 100 
    return thd

# ===================== Função de Processamento Principal =====================
@st.cache_data
def processar_arquivos(uploaded_files):
    colors = qualitative.Plotly
    metrics_data = {'Simulacao': [], 'CasoFalta': [], 'Local_m1': [], 'Cor': []}
    
    # Inicializa colunas do DataFrame
    for p in PONTOS_BASE:
        for var in ['I', 'V']:
            for fase in FASES:
                for m in METRICAS_TYPES:
                    metrics_data[f'{var}{p}_{m}_Fase{fase}'] = []

    spectrum_data = [] 
    line_figs = {}

    # --- 1. Inicializa Figuras de Linha (3 Fases x 2 Tipos x Pontos) ---
    for p in PONTOS_BASE:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            key_root = f'{var}{p.replace("_", "")}' # ex: I800
            
            for fase in FASES:
                # Figura Tempo
                fig_t_key = f'{key_root}_T_{fase}'
                line_figs[fig_t_key] = go.Figure(layout=go.Layout(
                    title=f'Sinal no Tempo: {label} ({var}_{p}) - Fase {fase}', 
                    xaxis_title='Tempo (s)', yaxis_title='Amplitude'))
                
                # Figura FFT
                fig_f_key = f'{key_root}_F_{fase}'
                line_figs[fig_f_key] = go.Figure(layout=go.Layout(
                    title=f'Espectro FFT: {label} ({var}_{p}) - Fase {fase}', 
                    xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)', 
                    xaxis=dict(range=[0, F_MAX_ANALISE/2], autorange=False))) # Zoom 1kHz
                
                # Marcadores
                for h in HARMONICOS_IMPARES:
                    line_figs[fig_f_key].add_vline(x=h*F_FUNDAMENTAL, line_width=0.5, line_dash="dot", line_color="rgba(128, 128, 128, 0.3)")
                    if h <= 9: 
                        line_figs[fig_f_key].add_annotation(x=h*F_FUNDAMENTAL, y=1, yref="paper", text=f"H{h}", showarrow=False, font_size=8, yshift=10)

    # --- 2. Loop Principal de Arquivos ---
    for i, file in enumerate(uploaded_files):
        matFile = file.name
        currentColor = colors[i % len(colors)]
        
        try:
            data = sio.loadmat(io.BytesIO(file.read()))
            
            plotTitle = matFile.replace('.mat', '').replace('__', ' - ').replace('_', ' ')
            try:
                parts = plotTitle.split(' - ', 1)
                simName = parts[0].strip()
                caseName = parts[1].strip() if len(parts) > 1 else 'Sem Falta'
            except:
                simName = plotTitle; caseName = "N/A"
            
            metrics_data['Simulacao'].append(simName)
            metrics_data['CasoFalta'].append(caseName)
            metrics_data['Cor'].append(currentColor)
            
            try: m1_loc = data['m1_location'][0, 0]; 
            except: m1_loc = 0;
            metrics_data['Local_m1'].append(m1_loc)
            
            struct_ts = data['ts'][0, 0]
            struct_fft = data['fft_data'][0, 0]
            
            # --- Processamento por Ponto ---
            for p in PONTOS_BASE:
                for var in ['I', 'V']:
                    vn = f"{var}_{p}"
                    clean_vn_fig = vn.replace('_', '')
                    
                    field_ts = f'ts_{vn}'   
                    field_f = f'f_{vn}'     
                    field_p1 = f'P1_{vn}'   

                    try:
                        # Extração
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y_all_phases = ts_data_struct['Data']
                        
                        f = struct_fft[field_f].flatten()
                        P1_all_phases = struct_fft[field_p1]
                        
                        # Correção de dimensões
                        if y_all_phases.shape[1] == 1: y_all_phases = np.tile(y_all_phases, (1, 3))
                        if P1_all_phases.shape[1] == 1: P1_all_phases = np.tile(P1_all_phases, (1, 3))
                        
                        # Loop por Fase (A, B, C)
                        for idx, fase in enumerate(FASES):
                            y_fase = y_all_phases[:, idx]
                            P1_fase = P1_all_phases[:, idx]
                            
                            # Plotagem
                            fig_t_key = f'{clean_vn_fig}_T_{fase}'
                            fig_f_key = f'{clean_vn_fig}_F_{fase}'
                            
                            line_figs[fig_t_key].add_trace(go.Scatter(x=t, y=y_fase, name=plotTitle, line=dict(color=currentColor)))
                            line_figs[fig_f_key].add_trace(go.Scatter(x=f, y=20*np.log10(P1_fase+1e-9), name=plotTitle, line=dict(color=currentColor)))

                            # Cálculo Métricas
                            pico = np.max(np.abs(y_fase))
                            thd = calculate_thd(f, P1_fase, F_FUNDAMENTAL, F_MAX_ANALISE)
                            h3 = get_harmonic_amplitude(f, P1_fase, 3)
                            h5 = get_harmonic_amplitude(f, P1_fase, 5)
                            h7 = get_harmonic_amplitude(f, P1_fase, 7)
                            
                            metrics_data[f'{var}{p}_Pico_Fase{fase}'].append(pico)
                            metrics_data[f'{var}{p}_THD_Fase{fase}'].append(thd)
                            metrics_data[f'{var}{p}_H3_Fase{fase}'].append(h3)
                            metrics_data[f'{var}{p}_H5_Fase{fase}'].append(h5)
                            metrics_data[f'{var}{p}_H7_Fase{fase}'].append(h7)

                        # Espectro (T2F - Fase A como representativa, ou adicionar lógica para todas)
                        if p == 'T2F' and var == 'I':
                            amps_A = [get_harmonic_amplitude(f, P1_all_phases[:,0], h) for h in [1]+HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': f"{plotTitle} (Fase A)", 'Amps': amps_A, 'Cor': currentColor})

                    except Exception:
                        for fase in FASES:
                            for m in METRICAS_TYPES:
                                metrics_data[f'{var}{p}_{m}_Fase{fase}'].append(np.nan)
        except Exception as e:
            st.error(f"Erro ao ler arquivo {matFile}: {e}")
            continue

    # --- Gráficos de Barra ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)
    
    if not df.empty:
        def create_bar(df_in, y_col, title, y_unit):
            fig = go.Figure(go.Bar(x=df_in['CasoFalta'], y=df_in[y_col], marker_color=df_in['Cor'], name=title))
            fig.update_layout(title=title, yaxis_title=y_unit, xaxis_tickangle=-45)
            return fig

        for p in PONTOS_BASE:
            for var in ['I', 'V']:
                unit = 'A' if var == 'I' else 'V'
                for fase in FASES:
                    for m in METRICAS_TYPES:
                        col_name = f'{var}{p}_{m}_Fase{fase}'
                        if col_name in df.columns:
                            m_label = "Pico" if m == "Pico" else "THD" if m == "THD" else f"{m}ª Harmônica"
                            bar_figs[f'Bar_{col_name}'] = create_bar(df, col_name, f'{m_label} {var} ({p}) - Fase {fase}', unit if m != 'THD' else '%')

        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title='Espectro Harmônico (I_T2F, Fase A)', barmode='group')
            bar_figs['Spectrum_Full'] = sp_fig

    return line_figs, bar_figs, df

# ===================== Interface =====================
uploaded_files = st.file_uploader("Selecione arquivos .mat", accept_multiple_files=True, type=['.mat'])

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs, df_metrics = processar_arquivos(tuple(uploaded_files))
    st.success("Concluído!")
    
    # Função para Extremos
    def show_extreme_metric(df, col, title, find_max=True):
        if col not in df.columns or df[col].isnull().all(): return
        try:
            idx = df[col].idxmax() if find_max else df[col].idxmin()
            label = f"Pior Caso ({'Máx' if find_max else 'Mín'}): {title}"
            val = df.loc[idx, col]
            case = df.loc[idx, 'CasoFalta']
            st.metric(label=label, value=f"{val:.2f}", delta=f"Caso: {case}", delta_color="off")
        except: pass

    # --- Abas ---
    tab_names = ["🏆 Extremos", "📉 Localização (m1)", "📊 Métricas", "🌊 Espectro"] + [f"📍 {p}" for p in PONTOS_BASE]
    tabs = st.tabs(tab_names)

    # --- Aba 0: Extremos ---
    with tabs[0]:
        fase_select = st.radio("Selecione a Fase para análise de extremos:", FASES, horizontal=True)
        st.subheader(f"Piores Casos (Fase {fase_select})")
        if not df_metrics.empty:
            cols = st.columns(3)
            with cols[0]: show_extreme_metric(df_metrics, f'I800_Pico_Fase{fase_select}', f"Pico Corrente I_800 (A)", True)
            with cols[1]: show_extreme_metric(df_metrics, f'I800_THD_Fase{fase_select}', f"THD Corrente I_800 (%)", True)
            with cols[2]: show_extreme_metric(df_metrics, f'V800_Pico_Fase{fase_select}', f"Tensão Mínima V_800 (V)", False)

    # --- Aba 1: Varredura m1 ---
    with tabs[1]:
        st.header("Análise de Localização de Falta (Varredura m1)")
        df_m1 = df_metrics[df_metrics['Local_m1'] > 0].copy()
        
        if df_meio_empty := df_m1.empty:
            st.warning("Nenhum arquivo de falta 'no meio' (m1 > 0) foi carregado.")
        else:
            c1, c2, c3 = st.columns(3)
            sim_choice = c1.selectbox("Filtrar por Simulação:", df_m1['Simulacao'].unique())
            fase_choice = c2.selectbox("Fase:", FASES)
            
            # Filtra colunas baseadas na fase escolhida
            metric_options = [col.replace(f'_Fase{fase_choice}', '') for col in df_m1.columns if f'Fase{fase_choice}' in col and ('Pico' in col or 'THD' in col)]
            metric_base = c3.selectbox("Métrica:", metric_options)
            metric_full = f"{metric_base}_Fase{fase_choice}"
            
            df_plot = df_m1[df_m1['Simulacao'] == sim_choice]
            fig_m1 = go.Figure()
            for case in df_plot['CasoFalta'].unique():
                d = df_plot[df_plot['CasoFalta'] == case].sort_values('Local_m1')
                fig_m1.add_trace(go.Scatter(x=d['Local_m1'], y=d[metric_full], mode='lines+markers', name=case))
            
            fig_m1.update_layout(xaxis_title="Localização (m1)", yaxis_title=metric_full)
            st.plotly_chart(fig_m1, use_container_width=True)

    # --- Aba 2: Métricas (Barras) ---
    with tabs[2]:
        fase_bar_select = st.radio("Visualizar Fase:", FASES, horizontal=True, key="bar_fase")
        st.markdown(f"### Comparação Quantitativa (Fase {fase_bar_select})")
        
        for p in PONTOS_BASE:
            with st.expander(f"Dados do Ponto {p}", expanded=(p=='T2F')):
                c1, c2, c3 = st.columns(3)
                if f'Bar_I{p}_Pico_Fase{fase_bar_select}' in bar_figs: 
                    c1.plotly_chart(bar_figs[f'Bar_I{p}_Pico_Fase{fase_bar_select}'], use_container_width=True)
                if f'Bar_I{p}_THD_Fase{fase_bar_select}' in bar_figs: 
                    c2.plotly_chart(bar_figs[f'Bar_I{p}_THD_Fase{fase_bar_select}'], use_container_width=True)
                if f'Bar_I{p}_H3_Fase{fase_bar_select}' in bar_figs: 
                    c3.plotly_chart(bar_figs[f'Bar_I{p}_H3_Fase{fase_bar_select}'], use_container_width=True)

    # --- Aba 3: Espectro ---
    with tabs[3]:
        if 'Spectrum_Full' in bar_figs: st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

    # --- Abas 4+: Pontos (Fase A, B, C dentro) ---
    for i, p in enumerate(PONTOS_BASE):
        with tabs[i+4]:
            clean_p = p.replace('_', '')
            
            # Sub-abas para as fases
            tab_A, tab_B, tab_C = st.tabs(["Fase A", "Fase B", "Fase C"])
            
            for sub_tab, fase in zip([tab_A, tab_B, tab_C], FASES):
                with sub_tab:
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(line_figs[f'I{clean_p}_T_{fase}'], use_container_width=True)
                    c2.plotly_chart(line_figs[f'I{clean_p}_F_{fase}'], use_container_width=True)
                    c3, c4 = st.columns(2)
                    c3.plotly_chart(line_figs[f'V{clean_p}_T_{fase}'], use_container_width=True)
                    c4.plotly_chart(line_figs[f'V{clean_p}_F_{fase}'], use_container_width=True)

    # Download
    st.sidebar.markdown("### 📥 Exportar")
    if st.sidebar.button("Gerar ZIP"):
        with st.spinner("Gerando imagens..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Salva CSV
                zf.writestr("metricas_completas.csv", df_metrics.to_csv(index=False).encode('utf-8'))
                
                # Salva Imagens (Todas as Fases)
                for name, fig in {**line_figs, **bar_figs}.items():
                    try:
                        fname = f"{name}.png" 
                        zf.writestr(fname, fig.to_image(format="png", width=1200, height=700))
                    except: pass
            st.sidebar.download_button("Baixar ZIP", zip_buffer.getvalue(), "resultados.zip", "application/zip")
else:
    st.info("Faça upload dos arquivos .mat.")