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
st.set_page_config(layout="wide", page_title="Analisador de Faltas T2F")
st.title("Analisador Avançado de Simulação de Faltas")

# ===================== Constantes =====================
F_FUNDAMENTAL = 60  # Hz
F_MAX_ANALISE = 2000 # Analisar THD até 2kHz
HARMONICOS_IMPARES = [h for h in range(3, int(F_MAX_ANALISE / F_FUNDAMENTAL) + 1, 2) if h <= 15]

# (CORRIGIDO) Lista completa de pontos base
PONTOS_BASE = ['800', 'T2F', '818_1', '818_2', '820', '822']
METRICAS_CALC = ['Pico', 'THD', 'H3', 'H5', 'H7']

# ===================== Funções Auxiliares =====================
@st.cache_data
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    """Encontra a amplitude de uma frequência específica."""
    if freq_array.size == 0: return 0
    target_freq = order * fund_freq
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]

@st.cache_data
def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    """Calcula THD."""
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
    
    # Dicionário para armazenar todas as métricas para o DataFrame
    metrics_data = {'Simulacao': [], 'CasoFalta': [], 'Local_m1': [], 'Cor': []}
    
    # Inicializa colunas do DataFrame
    for p in PONTOS_BASE:
        for var in ['I', 'V']:
            for fase in ['A', 'B', 'C']:
                for m in METRICAS_CALC:
                    metrics_data[f'{var}{p}_{m}_Fase{fase}'] = []

    spectrum_data = [] 
    line_figs = {}

    # Inicializa Figuras de Linha
    for p in PONTOS_BASE:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            key_root = f'{var}{p.replace("_", "")}' # ex: I800, VT2F, I8181
            
            line_figs[f'{key_root}_T'] = go.Figure(layout=go.Layout(
                title=f'Sinal no Tempo: {label} ({var}_{p}) - Fase A', 
                xaxis_title='Tempo (s)', yaxis_title='Amplitude'))
            
            line_figs[f'{key_root}_F'] = go.Figure(layout=go.Layout(
                title=f'Espectro FFT: {label} ({var}_{p}) - Fase A', 
                xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)', 
                xaxis=dict(range=[0, F_MAX_ANALISE/2], autorange=False))) # Zoom 1kHz
            
            for h in HARMONICOS_IMPARES:
                line_figs[f'{key_root}_F'].add_vline(x=h*F_FUNDAMENTAL, line_width=0.5, line_dash="dot", line_color="rgba(128, 128, 128, 0.3)")
                if h <= 9: line_figs[f'{key_root}_F'].add_annotation(x=h*F_FUNDAMENTAL, y=1, yref="paper", text=f"H{h}", showarrow=False, font_size=8, yshift=10)

    # --- Loop Principal de Arquivos ---
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
            
            try:
                m1_loc = data['m1_location'][0, 0]
                metrics_data['Local_m1'].append(m1_loc)
            except KeyError:
                metrics_data['Local_m1'].append(0) 
            
            struct_ts = data['ts'][0, 0]
            struct_fft = data['fft_data'][0, 0]
            
            # --- Processamento por Ponto de Medição ---
            for p in PONTOS_BASE: # ex: '818_1'
                for var in ['I', 'V']: # ex: 'I'
                    vn = f"{var}_{p}" # ex: I_818_1
                    clean_vn_fig = vn.replace('_', '') # ex: I8181 (para dict)
                    
                    field_ts = f'ts_{vn}'   
                    field_f = f'f_{vn}'     
                    field_p1 = f'P1_{vn}'   
                    
                    fig_key_root = clean_vn_fig 

                    try:
                        # (CORRIGIDO) Acesso à estrutura de dados
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y_all_phases = ts_data_struct['Data']
                        
                        f = struct_fft[field_f].flatten()
                        P1_all_phases = struct_fft[field_p1]
                        
                        if y_all_phases.shape[1] == 1: y_all_phases = np.tile(y_all_phases, (1, 3))
                        if P1_all_phases.shape[1] == 1: P1_all_phases = np.tile(P1_all_phases, (1, 3))
                        
                        # Plotagem (Apenas Fase A)
                        y_faseA = y_all_phases[:, 0]
                        P1_faseA = P1_all_phases[:, 0]
                        line_figs[f'{fig_key_root}_T'].add_trace(go.Scatter(x=t, y=y_faseA, name=plotTitle, line=dict(color=currentColor)))
                        line_figs[f'{fig_key_root}_F'].add_trace(go.Scatter(x=f, y=20*np.log10(P1_faseA+1e-9), name=plotTitle, line=dict(color=currentColor)))

                        # Cálculo de Métricas (Todas as 3 Fases)
                        for fase_idx, fase_nome in enumerate(['A', 'B', 'C']):
                            y_fase = y_all_phases[:, fase_idx]
                            P1_fase = P1_all_phases[:, fase_idx]
                            
                            pico = np.max(np.abs(y_fase))
                            thd = calculate_thd(f, P1_fase, fund_freq=F_FUNDAMENTAL, max_freq=F_MAX_ANALISE)
                            h3 = get_harmonic_amplitude(f, P1_fase, 3)
                            h5 = get_harmonic_amplitude(f, P1_fase, 5)
                            h7 = get_harmonic_amplitude(f, P1_fase, 7)
                            
                            metrics_data[f'{var}{p}_Pico_Fase{fase_nome}'].append(pico)
                            metrics_data[f'{var}{p}_THD_Fase{fase_nome}'].append(thd)
                            metrics_data[f'{var}{p}_H3_Fase{fase_nome}'].append(h3)
                            metrics_data[f'{var}{p}_H5_Fase{fase_nome}'].append(h5)
                            metrics_data[f'{var}{p}_H7_Fase{fase_nome}'].append(h7)

                        if p == 'T2F' and var == 'I':
                            amps_A = [get_harmonic_amplitude(f, P1_all_phases[:, 0], h) for h in [1] + HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': f"{plotTitle} (Fase A)", 'Amps': amps_A, 'Cor': currentColor})

                    except Exception:
                        for fase_nome in ['A', 'B', 'C']:
                            for m in METRICAS_CALC:
                                metrics_data[f'{var}{p}_{m}_Fase{fase_nome}'].append(np.nan)
        except Exception as e:
            st.error(f"Erro fatal ao ler {matFile}: {e}")
            keys = [k for k in data.keys() if not k.startswith('__')]
            st.warning(f"Claves encontradas: {keys}")
            continue

    # --- Criação dos Gráficos de Barra ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)
    
    if not df.empty:
        def create_bar(df_in, y_col, title_text, y_label):
            fig = go.Figure(go.Bar(x=df_in['Caso'], y=df_in[y_col], marker_color=df_in['Cor'], name=title_text))
            fig.update_layout(title=title_text, yaxis_title=y_label, xaxis_tickangle=-45)
            return fig

        for p in PONTOS_BASE:
            for var in ['I', 'V']:
                base = f'{var}{p}'
                unit = 'A' if var == 'I' else 'V'
                for m_type in METRICAS_CALC: 
                    metric_key = f'{base}_{m_type}_FaseA'
                    m_label = "Pico" if m_type == "Pico" else "THD" if m_type == "THD" else f"{m_type[1]}ª Harmônica"
                    bar_figs[f'Bar_{base}_{m_type}'] = create_bar(df, metric_key, f'{m_label} {var} ({p}) - Fase A', unit)

        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title=f'Espectro Harmônico (I_T2F, Fase A)', 
                                   yaxis_title='Amplitude (A)', barmode='group')
            bar_figs['Spectrum_Full'] = sp_fig

    return line_figs, bar_figs, df

# ===================== Interface =====================
uploaded_files = st.file_uploader(
    "Selecione arquivos .mat (gerados pelo script de varredura m1)",
    accept_multiple_files=True, type=['.mat']
)

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs, df_metrics = processar_arquivos(tuple(uploaded_files))

    st.success("Concluído!")
    
    # --- Função para a Aba de Extremos ---
    def show_extreme_metric(df, col, title, find_max=True):
        if col not in df.columns or df[col].isnull().all(): return
        try:
            if find_max:
                extreme_idx = df[col].idxmax()
                label = f"Pior Caso (Máx): {title}"
            else:
                extreme_idx = df[col].idxmin()
                label = f"Pior Caso (Mín): {title}"
            extreme_val = df.loc[extreme_idx, col]
            extreme_case = df.loc[extreme_idx, 'Caso']
            st.metric(label=label, value=f"{extreme_val:.2f}")
            st.caption(f"Caso: {extreme_case}")
        except Exception: pass

    # --- Definição das Abas ---
    tab_names = ["🏆 Análise de Extremos", "📉 Análise de Localização (m1)", "📊 Métricas (Barras)", "🌊 Espectro (I_T2F)"] + [f"📍 {p}" for p in PONTOS_BASE]
    tabs = st.tabs(tab_names)

    # --- Aba 0: Extremos ---
    with tabs[0]:
        st.header("Análise de Casos Extremos (Fase A)")
        if df_metrics.empty:
            st.error("Nenhum dado válido processado.")
        else:
            for p in PONTOS_BASE:
                with st.expander(f"Extremos do Ponto {p}"):
                    c1, c2, c3 = st.columns(3)
                    with c1: show_extreme_metric(df_metrics, f'I{p}_Pico_FaseA', f"Pico Corrente {p} (A)", find_max=True)
                    with c2: show_extreme_metric(df_metrics, f'I{p}_THD_FaseA', f"THD Corrente {p} (%)", find_max=True)
                    with c3: show_extreme_metric(df_metrics, f'V{p}_Pico_FaseA', f"Tensão Mínima {p} (V)", find_max=False)

    # --- Aba 1: Varredura m1 ---
    with tabs[1]:
        st.header("Análise de Localização de Falta (Varredura m1)")
        df_meio = df_metrics[df_metrics['Local_m1'] > 0].copy()
        
        if df_meio.empty:
            st.warning("Nenhum arquivo de falta 'no meio' (m1 > 0) foi carregado.")
        else:
            c1, c2 = st.columns(2)
            sim_options = df_meio['Simulacao'].unique()
            sim_choice = c1.selectbox("Filtrar por Simulação:", sim_options)
            
            metric_options = [col for col in df_meio.columns if 'Pico' in col or 'THD' in col]
            metric_choice = c2.selectbox("Selecione a Métrica para Analisar:", metric_options)
            
            df_plot_sim = df_meio[df_meio['Simulacao'] == sim_choice]
            
            fig_m1 = go.Figure()
            for case_name in df_plot_sim['CasoFalta'].unique():
                df_case = df_plot_sim[df_plot_sim['CasoFalta'] == case_name].sort_values(by='Local_m1')
                if not df_case.empty:
                    fig_m1.add_trace(go.Scatter(
                        x=df_case['Local_m1'], 
                        y=df_case[metric_choice],
                        mode='lines+markers',
                        name=case_name
                    ))
            
            fig_m1.update_layout(
                title=f"{metric_choice} vs. Localização da Falta (m1)",
                xaxis_title="Localização da Falta (m1, 0.01 a 0.99)",
                yaxis_title=metric_choice,
                legend_title="Tipo de Falta"
            )
            st.plotly_chart(fig_m1, use_container_width=True)

    # --- Aba 2: Métricas (Barras) ---
    with tabs[2]:
        st.markdown("### Comparação Quantitativa (Fase A)")
        for p in PONTOS_BASE:
            with st.expander(f"Dados do Ponto {p}", expanded=(p=='T2F')):
                c1, c2, c3 = st.columns(3)
                c1.plotly_chart(bar_figs[f'Bar_I{p}_Pico'], use_container_width=True)
                c2.plotly_chart(bar_figs[f'Bar_I{p}_THD'], use_container_width=True)
                c3.plotly_chart(bar_figs[f'Bar_I{p}_H3'], use_container_width=True)

    # --- Aba 3: Espectro (I_T2F) ---
    with tabs[3]:
        if 'Spectrum_Full' in bar_figs:
            st.markdown(f"#### Espectro Harmônico (I_T2F) - H1 até ~H{HARMONICOS_IMPARES[-1]}")
            st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

    # --- Abas 4+: Formas de Onda por Ponto ---
    for i, p in enumerate(PONTOS_BASE):
        with tabs[i+4]: # Começa na aba de índice 4
            st.markdown(f"### Análise Detalhada: Ponto {p}")
            key_I = f"I{p.replace('_', '')}"
            key_V = f"V{p.replace('_', '')}"
            c1, c2 = st.columns(2)
            c1.plotly_chart(line_figs[f'{key_I}_T'], use_container_width=True)
            c2.plotly_chart(line_figs[f'{key_I}_F'], use_container_width=True)
            c3, c4 = st.columns(2)
            c3.plotly_chart(line_figs[f'{key_V}_T'], use_container_width=True)
            c4.plotly_chart(line_figs[f'{key_V}_F'], use_container_width=True)

    # --- Download ---
    st.sidebar.markdown("### 📥 Exportar")
    if st.sidebar.button("Gerar ZIP"):
        with st.spinner("Gerando imagens..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                all_figs = {**line_figs, **bar_figs}
                for name, fig in all_figs.items():
                    try:
                        fname = f"{name}.png" 
                        zf.writestr(fname, fig.to_image(format="png", width=1200, height=700))
                    except: pass
            st.sidebar.download_button("Baixar ZIP", zip_buffer.getvalue(), "resultados.zip", "application/zip")
else:
    st.info("Faça upload dos arquivos .mat (gerados pelo script MATLAB).")