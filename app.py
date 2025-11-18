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
st.set_page_config(layout="wide", page_title="Analisador T2F")
st.title("Analisador Avançado de Qualidade de Energia (MATLAB/Python)")

# ===================== Constantes =====================
F_FUNDAMENTAL = 60  # Hz
PLOT_XLIM_FFT = [0, 1000]
HARMONICOS_IMPARES = [3, 5, 7, 9, 11, 13, 15]
# Lista exata de pontos conforme seu script MATLAB
PONTOS_BASE = ['800', 'T2F', '818_1', '818_2', '820', '822']
# Definido GLOBALMENTE para evitar erro de escopo
METRICAS_TYPES = ['Pico', 'THD', 'H3', 'H5', 'H7'] 

# ===================== Funções Auxiliares =====================
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    if freq_array.size == 0: return 0
    target_freq = order * fund_freq
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]

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
    line_figs = {}

    # Inicializa Figuras de Linha
    for p in PONTOS_BASE:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            # Remove underscores para chaves limpas (ex: I8181)
            clean_p = p.replace('_', '')
            key_root = f'{var}{clean_p}'
            
            line_figs[f'{key_root}_T'] = go.Figure(layout=go.Layout(
                title=f'Sinal no Tempo: {label} ({var}_{p}) - Fase A', 
                xaxis_title='Tempo (s)', yaxis_title='Amplitude'))
            
            line_figs[f'{key_root}_F'] = go.Figure(layout=go.Layout(
                title=f'Espectro FFT: {label} ({var}_{p}) - Fase A', 
                xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)', 
                xaxis=dict(range=[0, F_MAX_ANALISE/2], autorange=False)))
            
            for h in HARMONICOS_IMPARES:
                line_figs[f'{key_root}_F'].add_vline(x=h*F_FUNDAMENTAL, line_width=0.5, line_dash="dot", line_color="rgba(128, 128, 128, 0.3)")

    # --- Loop Principal ---
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
            
            # Acessa Structs Principais com segurança
            # No MATLAB: tosave.ts e tosave.fft_data
            # No Python (scipy): data['ts'][0,0]
            if 'ts' not in data or 'fft_data' not in data:
                raise ValueError("Estrutura do arquivo inválida (faltam 'ts' ou 'fft_data')")
                
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
                        # O scipy.io carrega structs como arrays numpy de dimensão 0 (void)
                        # Precisamos checar se o campo existe nos nomes (dtype.names)
                        if field_ts not in struct_ts.dtype.names:
                            raise ValueError(f"Campo {field_ts} não encontrado")

                        # Acessa dados aninhados: struct_ts[campo][0,0]['Data']
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y_all_phases = ts_data_struct['Data']
                        
                        f = struct_fft[field_f].flatten()
                        P1_all_phases = struct_fft[field_p1]
                        
                        # Corrige dimensões (1 fase -> 3 fases)
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
                            thd = calculate_thd(f, P1_fase, F_FUNDAMENTAL, 2000)
                            h3 = get_harmonic_amplitude(f, P1_fase, 3)
                            h5 = get_harmonic_amplitude(f, P1_fase, 5)
                            h7 = get_harmonic_amplitude(f, P1_fase, 7)
                            
                            metrics_data[f'{var}{p}_{"Pico"}_Fase{fase_nome}'].append(pico)
                            metrics_data[f'{var}{p}_{"THD"}_Fase{fase_nome}'].append(thd)
                            metrics_data[f'{var}{p}_{"H3"}_Fase{fase_nome}'].append(h3)
                            metrics_data[f'{var}{p}_{"H5"}_Fase{fase_nome}'].append(h5)
                            metrics_data[f'{var}{p}_{"H7"}_Fase{fase_nome}'].append(h7)

                        # Espectro (T2F)
                        if p == 'T2F' and var == 'I':
                            amps = [get_harmonic_amplitude(f, P1_all_phases[:,0], h) for h in [1]+HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': f"{plotTitle}", 'Amps': amps, 'Cor': currentColor})

                    except Exception:
                        # Preenche com NaN se falhar a leitura deste ponto
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
            fig = go.Figure(go.Bar(x=df_in['CasoFalta'], y=df_in[y_col], marker_color=df_in['Cor'], name=title))
            fig.update_layout(title=title, yaxis_title=y_unit, xaxis_tickangle=-45)
            return fig

        for p in PONTOS_BASE:
            for var in ['I', 'V']:
                unit = 'A' if var == 'I' else 'V'
                for m in METRICAS_TYPES:
                    col_name = f'{var}{p}_{m}_FaseA' # Apenas Fase A para visualização rápida
                    if col_name in df.columns:
                        bar_figs[f'Bar_{col_name}'] = create_bar(df, col_name, f'{m} {var} ({p}) - Fase A', unit if m != 'THD' else '%')

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
    
    tabs = st.tabs(["🏆 Extremos", "📉 Localização (m1)", "📊 Métricas", "🌊 Espectro"] + [f"📍 {p}" for p in PONTOS_BASE])

    # Aba Extremos
    with tabs[0]:
        st.subheader("Piores Casos (Fase A)")
        if not df_metrics.empty:
            cols = st.columns(3)
            with cols[0]: 
                idx = df_metrics['I800_Pico_FaseA'].idxmax()
                st.metric("Max I_800 Pico", f"{df_metrics.loc[idx, 'I800_Pico_FaseA']:.2f} A", df_metrics.loc[idx, 'CasoFalta'])
            with cols[1]:
                idx = df_metrics['I800_THD_FaseA'].idxmax()
                st.metric("Max I_800 THD", f"{df_metrics.loc[idx, 'I800_THD_FaseA']:.2f} %", df_metrics.loc[idx, 'CasoFalta'])
            with cols[2]:
                idx = df_metrics['V800_Pico_FaseA'].idxmin()
                st.metric("Min V_800 Tensão", f"{df_metrics.loc[idx, 'V800_Pico_FaseA']:.2f} V", df_metrics.loc[idx, 'CasoFalta'])

    # Aba m1
    with tabs[1]:
        st.subheader("Análise por Localização (m1)")
        df_m1 = df_metrics[df_metrics['Local_m1'] > 0]
        if not df_m1.empty:
            metric = st.selectbox("Métrica", [c for c in df_m1.columns if 'Pico' in c or 'THD' in c])
            fig = go.Figure()
            for sim in df_m1['Simulacao'].unique():
                d = df_m1[df_m1['Simulacao'] == sim].sort_values('Local_m1')
                fig.add_trace(go.Scatter(x=d['Local_m1'], y=d[metric], mode='lines+markers', name=sim))
            st.plotly_chart(fig, use_container_width=True)

    # Aba Métricas
    with tabs[2]:
        for p in PONTOS_BASE:
            with st.expander(f"Ponto {p}", expanded=(p=='T2F')):
                c1, c2 = st.columns(2)
                if f'Bar_I{p}_Pico_FaseA' in bar_figs: c1.plotly_chart(bar_figs[f'Bar_I{p}_Pico_FaseA'], use_container_width=True)
                if f'Bar_I{p}_THD_FaseA' in bar_figs: c2.plotly_chart(bar_figs[f'Bar_I{p}_THD_FaseA'], use_container_width=True)

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

else:
    st.info("Carregue os arquivos .mat")