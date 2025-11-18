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
st.set_page_config(layout="wide", page_title="Analisador de Harmônicas T2F")
st.title("Analisador Avançado de Qualidade de Energia (5 Pontos)")

# ===================== Constantes =====================
F_FUNDAMENTAL = 60  # Hz
F_MAX_ANALISE = 2000 # Limite para cálculo de THD (2 kHz)
PLOT_XLIM_FFT = [0, 1000] # Limite visual inicial dos gráficos (1 kHz)

# Gera harmônicas ímpares até o limite de análise
HARMONICOS_IMPARES = [h for h in range(3, int(F_MAX_ANALISE / F_FUNDAMENTAL) + 1, 2) if h <= 15] 

# Lista dos pontos de medição conforme salvos no MATLAB
PONTOS_BASE = ['800', 'T2F', '818_1', '818_2', '820', '822']
METRICAS_TYPES = ['Pico', 'THD', 'H3', 'H5', 'H7'] 

# ===================== Funções Auxiliares =====================
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    """Encontra a amplitude de uma frequência específica."""
    if freq_array.size == 0: return 0
    target_freq = order * fund_freq
    # Encontra o índice mais próximo da frequência alvo
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]

def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    """Calcula THD (Total Harmonic Distortion)."""
    if freq_array.size == 0: return 0
    amp_h1 = get_harmonic_amplitude(freq_array, amp_array, 1, fund_freq)
    if amp_h1 == 0: return 0
    
    sum_squares = 0
    for order in range(2, int(max_freq/fund_freq)):
        amp_hn = get_harmonic_amplitude(freq_array, amp_array, order, fund_freq)
        sum_squares += amp_hn ** 2
        
    thd = (np.sqrt(sum_squares) / amp_h1) * 100 
    return thd

# ===================== Função de Processamento =====================
@st.cache_data
def processar_arquivos(uploaded_files):
    colors = qualitative.Plotly
    metrics_data = {'Simulacao': [], 'CasoFalta': [], 'Local_m1': [], 'Cor': []}
    
    # Inicializa colunas do DataFrame
    for p in PONTOS_BASE:
        for var in ['I', 'V']:
            for fase in ['A', 'B', 'C']:
                for m in METRICAS_TYPES:
                    metrics_data[f'{var}{p}_{m}_Fase{fase}'] = []

    spectrum_data = [] 

    # --- 1. Inicializa Figuras de Linha ---
    line_figs = {}
    for p in PONTOS_BASE:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            # Chave limpa para o dicionário de figuras (ex: I8181)
            key_root = f'{var}{p.replace("_", "")}'
            
            # Figura Tempo
            line_figs[f'{key_root}_T'] = go.Figure(layout=go.Layout(
                title=f'Sinal no Tempo: {label} ({var}_{p}) - Fase A', 
                xaxis_title='Tempo (s)', yaxis_title='Amplitude'))
            
            # Figura FFT
            # Usa PLOT_XLIM_FFT para o visual inicial
            line_figs[f'{key_root}_F'] = go.Figure(layout=go.Layout(
                title=f'Espectro FFT: {label} ({var}_{p}) - Fase A', 
                xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)', 
                xaxis=dict(range=PLOT_XLIM_FFT, autorange=False)))
            
            # Marcadores de harmônicas
            for h in HARMONICOS_IMPARES:
                line_figs[f'{key_root}_F'].add_vline(x=h*F_FUNDAMENTAL, line_width=0.5, line_dash="dot", line_color="rgba(128, 128, 128, 0.3)")
                if h <= 9: 
                    line_figs[f'{key_root}_F'].add_annotation(
                        x=h*F_FUNDAMENTAL, y=1, yref="paper", text=f"H{h}", 
                        showarrow=False, font=dict(size=8, color="gray"), yshift=10
                    )

    # --- 2. Loop Principal ---
    for i, file in enumerate(uploaded_files):
        matFile = file.name
        currentColor = colors[i % len(colors)]
        
        try:
            data = sio.loadmat(io.BytesIO(file.read()))
            
            # Processa Nome
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
            
            # Processa m1
            try: m1_loc = data['m1_location'][0, 0]; 
            except: m1_loc = 0;
            metrics_data['Local_m1'].append(m1_loc)
            
            # Acessa Structs Principais
            if 'ts' not in data or 'fft_data' not in data:
                raise ValueError("Estrutura inválida (faltam 'ts' ou 'fft_data')")
                
            struct_ts = data['ts'][0, 0]
            struct_fft = data['fft_data'][0, 0]
            
            # --- Processamento por Ponto ---
            for p in PONTOS_BASE:
                for var in ['I', 'V']:
                    # Constrói nomes conforme salvos no MATLAB (ex: ts_I_818_1)
                    vn = f"{var}_{p}"
                    field_ts = f'ts_{vn}'
                    field_f = f'f_{vn}'
                    field_p1 = f'P1_{vn}'
                    
                    # Chave interna para figuras (sem underscore)
                    clean_p = p.replace('_', '')
                    fig_key_root = f'{var}{clean_p}'

                    try:
                        # Tenta ler o campo dentro do struct
                        if field_ts not in struct_ts.dtype.names:
                            raise ValueError(f"Campo {field_ts} não encontrado")

                        # Acessa dados
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y_all_phases = ts_data_struct['Data']
                        
                        f = struct_fft[field_f].flatten()
                        P1_all_phases = struct_fft[field_p1]
                        
                        # Corrige dimensões
                        if y_all_phases.shape[1] == 1: y_all_phases = np.tile(y_all_phases, (1, 3))
                        if P1_all_phases.shape[1] == 1: P1_all_phases = np.tile(P1_all_phases, (1, 3))
                        
                        # Plotagem (Fase A)
                        line_figs[f'{fig_key_root}_T'].add_trace(go.Scatter(x=t, y=y_all_phases[:,0], name=plotTitle, line=dict(color=currentColor)))
                        line_figs[f'{fig_key_root}_F'].add_trace(go.Scatter(x=f, y=20*np.log10(P1_all_phases[:,0]+1e-9), name=plotTitle, line=dict(color=currentColor)))

                        # Cálculo Métricas
                        for fase_idx, fase_nome in enumerate(['A', 'B', 'C']):
                            y_fase = y_all_phases[:, fase_idx]
                            P1_fase = P1_all_phases[:, fase_idx]
                            
                            pico = np.max(np.abs(y_fase))
                            # Usa F_MAX_ANALISE para o cálculo do THD
                            thd = calculate_thd(f, P1_fase, F_FUNDAMENTAL, F_MAX_ANALISE)
                            h3 = get_harmonic_amplitude(f, P1_fase, 3)
                            h5 = get_harmonic_amplitude(f, P1_fase, 5)
                            h7 = get_harmonic_amplitude(f, P1_fase, 7)
                            
                            metrics_data[f'{var}{p}_Pico_Fase{fase_nome}'].append(pico)
                            metrics_data[f'{var}{p}_THD_Fase{fase_nome}'].append(thd)
                            metrics_data[f'{var}{p}_H3_Fase{fase_nome}'].append(h3)
                            metrics_data[f'{var}{p}_H5_Fase{fase_nome}'].append(h5)
                            metrics_data[f'{var}{p}_H7_Fase{fase_nome}'].append(h7)

                        # Espectro (T2F)
                        if p == 'T2F' and var == 'I':
                            amps = [get_harmonic_amplitude(f, P1_all_phases[:,0], h) for h in [1]+HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': f"{plotTitle}", 'Amps': amps, 'Cor': currentColor})

                    except Exception:
                        # Preenche com NaN se falhar
                        for fase_nome in ['A', 'B', 'C']:
                            for m in METRICAS_TYPES:
                                metrics_data[f'{var}{p}_{m}_Fase{fase_nome}'].append(np.nan)

        except Exception as e:
            st.error(f"Erro ao ler arquivo {matFile}: {e}")
            continue

    # --- Gráficos de Barra ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)
    
    if not df.empty:
        def create_bar(df_in, y_col, title, y_unit):
            # Usa 'CasoFalta' para o eixo X
            fig = go.Figure(go.Bar(x=df_in['CasoFalta'], y=df_in[y_col], marker_color=df_in['Cor'], name=title))
            fig.update_layout(title=title, yaxis_title=y_unit, xaxis_tickangle=-45)
            return fig

        for p in PONTOS_BASE:
            for var in ['I', 'V']:
                unit = 'A' if var == 'I' else 'V'
                for m in METRICAS_TYPES:
                    col_name = f'{var}{p}_{m}_FaseA' # Fase A por padrão nos gráficos
                    if col_name in df.columns:
                        m_label = "Pico" if m == "Pico" else "THD" if m == "THD" else f"{m}ª Harmônica" # Corrigido display de H3, H5
                        bar_figs[f'Bar_{col_name}'] = create_bar(df, col_name, f'{m_label} {var} ({p}) - Fase A', unit if m != 'THD' else '%')

        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title='Espectro Harmônico (I_T2F)', barmode='group')
            bar_figs['Spectrum_Full'] = sp_fig

    return line_figs, bar_figs, df

# ===================== Interface =====================
uploaded_files = st.file_uploader("Selecione arquivos .mat", accept_multiple_files=True, type=['.mat'])

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs, df_metrics = processar_arquivos(tuple(uploaded_files))
    st.success("Concluído!")
    
    tab_names = ["🏆 Extremos", "📉 Localização (m1)", "📊 Métricas", "🌊 Espectro"] + [f"📍 {p}" for p in PONTOS_BASE]
    tabs = st.tabs(tab_names)

    # Aba Extremos
    with tabs[0]:
        st.subheader("Piores Casos (Fase A)")
        def show_extreme_metric(df, col, title, find_max=True):
            if col not in df.columns or df[col].isnull().all(): return
            try:
                idx = df[col].idxmax() if find_max else df[col].idxmin()
                label = f"Pior Caso ({'Máx' if find_max else 'Mín'}): {title}"
                val = df.loc[idx, col]
                case = df.loc[idx, 'CasoFalta']
                st.metric(label=label, value=f"{val:.2f}", delta=f"Caso: {case}", delta_color="off")
            except: pass

        if not df_metrics.empty:
            cols = st.columns(3)
            with cols[0]: show_extreme_metric(df_metrics, 'I800_Pico_FaseA', "Pico Corrente I_800 (A)", True)
            with cols[1]: show_extreme_metric(df_metrics, 'I800_THD_FaseA', "THD Corrente I_800 (%)", True)
            with cols[2]: show_extreme_metric(df_metrics, 'V800_Pico_FaseA', "Tensão Mínima V_800 (V)", False)

    # Aba m1
    with tabs[1]:
        st.subheader("Análise por Localização (m1)")
        df_m1 = df_metrics[df_metrics['Local_m1'] > 0]
        if not df_m1.empty:
            c1, c2 = st.columns(2)
            sim_choice = c1.selectbox("Filtrar por Simulação:", df_m1['Simulacao'].unique())
            # Filtra apenas colunas numéricas relevantes
            metric_cols = [c for c in df_m1.columns if ('Pico' in c or 'THD' in c) and 'FaseA' in c]
            metric_choice = c2.selectbox("Métrica", metric_cols)
            
            df_plot = df_m1[df_m1['Simulacao'] == sim_choice]
            fig_m1 = go.Figure()
            for case in df_plot['CasoFalta'].unique():
                d = df_plot[df_plot['CasoFalta'] == case].sort_values('Local_m1')
                fig_m1.add_trace(go.Scatter(x=d['Local_m1'], y=d[metric_choice], mode='lines+markers', name=case))
            
            fig_m1.update_layout(xaxis_title="Localização (m1)", yaxis_title=metric_choice)
            st.plotly_chart(fig_m1, use_container_width=True)
        else:
            st.info("Nenhum caso com variação de m1 encontrado.")

    # Aba Métricas
    with tabs[2]:
        for p in PONTOS_BASE:
            with st.expander(f"Ponto {p}", expanded=(p=='T2F')):
                c1, c2, c3 = st.columns(3)
                if f'Bar_I{p}_Pico_FaseA' in bar_figs: c1.plotly_chart(bar_figs[f'Bar_I{p}_Pico_FaseA'], use_container_width=True)
                if f'Bar_I{p}_THD_FaseA' in bar_figs: c2.plotly_chart(bar_figs[f'Bar_I{p}_THD_FaseA'], use_container_width=True)
                if f'Bar_I{p}_H3_FaseA' in bar_figs: c3.plotly_chart(bar_figs[f'Bar_I{p}_H3_FaseA'], use_container_width=True)

    # Aba Espectro
    with tabs[3]:
        if 'Spectrum_Full' in bar_figs: st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

    # Abas Pontos
    for i, p in enumerate(PONTOS_BASE):
        with tabs[i+4]:
            clean_p = p.replace('_', '')
            c1, c2 = st.columns(2)
            st.plotly_chart(line_figs[f'I{clean_p}_T'], use_container_width=True)
            st.plotly_chart(line_figs[f'I{clean_p}_F'], use_container_width=True)
            st.plotly_chart(line_figs[f'V{clean_p}_T'], use_container_width=True)
            st.plotly_chart(line_figs[f'V{clean_p}_F'], use_container_width=True)

    # Download
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
                # Salva CSV
                zf.writestr("metricas_completas.csv", df_metrics.to_csv(index=False).encode('utf-8'))
                
            st.sidebar.download_button("Baixar ZIP", zip_buffer.getvalue(), "resultados.zip", "application/zip")
else:
    st.info("Faça upload dos arquivos .mat.")