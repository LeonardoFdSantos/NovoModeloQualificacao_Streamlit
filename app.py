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
st.set_page_config(layout="wide", page_title="Analisador Trifásico T2F")
st.title("Analisador de Qualidade de Energia (Visão Trifásica Completa)")

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

def aplicar_estilo_grafico(fig):
    """Aplica um estilo padrão com legenda embaixo para economizar espaço lateral."""
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3, # Empurra a legenda para baixo do eixo X
            xanchor="center",
            x=0.5
        ),
        margin=dict(l=40, r=20, t=40, b=80), # Margem inferior maior para a legenda
        hovermode="x unified"
    )
    return fig

# ===================== Função de Processamento =====================
@st.cache_data
def processar_arquivos(uploaded_files):
    colors = qualitative.Plotly
    metrics_data = {'Simulacao': [], 'CasoFalta': [], 'Local_m1': [], 'Cor': []}
    
    # Inicializa colunas
    for p in PONTOS_BASE:
        for var in ['I', 'V']:
            for fase in FASES:
                for m in METRICAS_TYPES:
                    metrics_data[f'{var}{p}_{m}_Fase{fase}'] = []

    spectrum_data = [] 
    line_figs = {}

    # --- 1. Inicializa Figuras (Uma para cada Fase de cada Ponto) ---
    for p in PONTOS_BASE:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            key_root = f'{var}{p.replace("_", "")}'
            
            for fase in FASES:
                # Figura Tempo
                fig_t_key = f'{key_root}_T_{fase}'
                line_figs[fig_t_key] = go.Figure(layout=go.Layout(
                    title=f'Tempo: {var}_{p} (Fase {fase})', 
                    xaxis_title='Tempo (s)', yaxis_title='Amplitude'))
                aplicar_estilo_grafico(line_figs[fig_t_key])
                
                # Figura FFT
                fig_f_key = f'{key_root}_F_{fase}'
                line_figs[fig_f_key] = go.Figure(layout=go.Layout(
                    title=f'FFT: {var}_{p} (Fase {fase})', 
                    xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)', 
                    xaxis=dict(range=[0, F_MAX_ANALISE/2], autorange=False)))
                aplicar_estilo_grafico(line_figs[fig_f_key])
                
                for h in HARMONICOS_IMPARES:
                    line_figs[fig_f_key].add_vline(x=h*F_FUNDAMENTAL, line_width=0.5, line_dash="dot", line_color="gray", opacity=0.3)

    # --- 2. Loop Arquivos ---
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
            
            for p in PONTOS_BASE:
                for var in ['I', 'V']:
                    vn = f"{var}_{p}"
                    clean_vn_fig = vn.replace('_', '')
                    
                    field_ts = f'ts_{vn}'; field_f = f'f_{vn}'; field_p1 = f'P1_{vn}'   

                    try:
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y_all = ts_data_struct['Data']
                        f = struct_fft[field_f].flatten()
                        P1_all = struct_fft[field_p1]
                        
                        if y_all.shape[1] == 1: y_all = np.tile(y_all, (1, 3))
                        if P1_all.shape[1] == 1: P1_all = np.tile(P1_all, (1, 3))
                        
                        for idx, fase in enumerate(FASES):
                            y_f = y_all[:, idx]
                            P1_f = P1_all[:, idx]
                            
                            # Plotagem
                            line_figs[f'{clean_vn_fig}_T_{fase}'].add_trace(go.Scatter(x=t, y=y_f, name=plotTitle, line=dict(color=currentColor), showlegend=(idx==0)))
                            line_figs[f'{clean_vn_fig}_F_{fase}'].add_trace(go.Scatter(x=f, y=20*np.log10(P1_f+1e-9), name=plotTitle, line=dict(color=currentColor), showlegend=(idx==0)))

                            # Métricas
                            metrics_data[f'{var}{p}_Pico_Fase{fase}'].append(np.max(np.abs(y_f)))
                            metrics_data[f'{var}{p}_THD_Fase{fase}'].append(calculate_thd(f, P1_f, F_FUNDAMENTAL, F_MAX_ANALISE))
                            metrics_data[f'{var}{p}_H3_Fase{fase}'].append(get_harmonic_amplitude(f, P1_f, 3))
                            metrics_data[f'{var}{p}_H5_Fase{fase}'].append(get_harmonic_amplitude(f, P1_f, 5))
                            metrics_data[f'{var}{p}_H7_Fase{fase}'].append(get_harmonic_amplitude(f, P1_f, 7))

                        if p == 'T2F' and var == 'I':
                            amps_A = [get_harmonic_amplitude(f, P1_all[:,0], h) for h in [1]+HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': f"{plotTitle} (Fase A)", 'Amps': amps_A, 'Cor': currentColor})

                    except Exception:
                        for fase in FASES:
                            for m in METRICAS_TYPES: metrics_data[f'{var}{p}_{m}_Fase{fase}'].append(np.nan)
        except Exception as e:
            st.error(f"Erro ao ler {matFile}: {e}")
            continue

    # --- 3. Gráficos de Barra ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)
    
    if not df.empty:
        def create_bar(df_in, y_col, title, y_unit):
            fig = go.Figure(go.Bar(x=df_in['CasoFalta'], y=df_in[y_col], marker_color=df_in['Cor'], name=title))
            fig.update_layout(title=title, yaxis_title=y_unit, xaxis_tickangle=-45)
            return aplicar_estilo_grafico(fig)

        for p in PONTOS_BASE:
            for var in ['I', 'V']:
                unit = 'A' if var == 'I' else 'V'
                for fase in FASES:
                    for m in METRICAS_TYPES:
                        col = f'{var}{p}_{m}_Fase{fase}'
                        if col in df.columns:
                            # Nome curto para caber no título
                            short_m = "Pico" if m == "Pico" else "THD" if m == "THD" else f"H{m[1]}"
                            bar_figs[f'Bar_{col}'] = create_bar(df, col, f'{short_m} {var}{p} ({fase})', unit if m != 'THD' else '%')

        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title='Espectro Harmônico (I_T2F - Fase A)', barmode='group')
            aplicar_estilo_grafico(sp_fig)
            bar_figs['Spectrum_Full'] = sp_fig

    return line_figs, bar_figs, df

# ===================== Interface =====================
uploaded_files = st.file_uploader("Selecione arquivos .mat", accept_multiple_files=True, type=['.mat'])

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs, df_metrics = processar_arquivos(tuple(uploaded_files))
    st.success("Concluído!")
    
    tab_names = ["📊 Métricas", "🌊 Espectro", "🏆 Extremos"] + [f"📍 {p}" for p in PONTOS_BASE]
    tabs = st.tabs(tab_names)

    # --- Aba 1: Métricas (Dashboard com Abas de Fase) ---
    with tabs[0]:
        st.markdown("### Comparação Quantitativa")
        
        # Cria sub-abas para as fases DENTRO das métricas para limpar a tela
        subtab_A, subtab_B, subtab_C = st.tabs(["Fase A", "Fase B", "Fase C"])
        
        for fase, subtab in zip(FASES, [subtab_A, subtab_B, subtab_C]):
            with subtab:
                for p in PONTOS_BASE:
                    with st.expander(f"Dados do Ponto {p} - Fase {fase}", expanded=(p=='T2F')):
                        # Layout de 2 colunas para não ficar espremido
                        c1, c2 = st.columns(2)
                        
                        # Linha 1: Corrente
                        if f'Bar_I{p}_Pico_Fase{fase}' in bar_figs: 
                            c1.plotly_chart(bar_figs[f'Bar_I{p}_Pico_Fase{fase}'], use_container_width=True)
                        if f'Bar_I{p}_THD_Fase{fase}' in bar_figs: 
                            c2.plotly_chart(bar_figs[f'Bar_I{p}_THD_Fase{fase}'], use_container_width=True)
                        
                        # Linha 2: Harmônicas Corrente
                        c3, c4 = st.columns(2)
                        if f'Bar_I{p}_H3_Fase{fase}' in bar_figs:
                            c3.plotly_chart(bar_figs[f'Bar_I{p}_H3_Fase{fase}'], use_container_width=True)
                        if f'Bar_I{p}_H5_Fase{fase}' in bar_figs:
                            c4.plotly_chart(bar_figs[f'Bar_I{p}_H5_Fase{fase}'], use_container_width=True)

                        st.markdown("---")
                        
                        # Linha 3: Tensão
                        c5, c6 = st.columns(2)
                        if f'Bar_V{p}_Pico_Fase{fase}' in bar_figs:
                            c5.plotly_chart(bar_figs[f'Bar_V{p}_Pico_Fase{fase}'], use_container_width=True)
                        if f'Bar_V{p}_THD_Fase{fase}' in bar_figs:
                            c6.plotly_chart(bar_figs[f'Bar_V{p}_THD_Fase{fase}'], use_container_width=True)

    # --- Aba 2: Espectro ---
    with tabs[1]:
        if 'Spectrum_Full' in bar_figs: st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

    # --- Aba 3: Extremos ---
    with tabs[2]:
        st.subheader("Resumo de Extremos (Todas as Fases)")
        if not df_metrics.empty:
            # Itera sobre cada fase para mostrar os extremos
            for fase in FASES:
                st.markdown(f"#### Fase {fase}")
                cols = st.columns(4)
                
                # Ponto 800
                idx = df_metrics[f'I800_Pico_Fase{fase}'].idxmax()
                cols[0].metric(f"Máx I_800", f"{df_metrics.loc[idx, f'I800_Pico_Fase{fase}']:.1f} A", df_metrics.loc[idx, 'CasoFalta'])
                
                idx = df_metrics[f'I800_THD_Fase{fase}'].idxmax()
                cols[1].metric(f"Máx THD I_800", f"{df_metrics.loc[idx, f'I800_THD_Fase{fase}']:.1f} %", df_metrics.loc[idx, 'CasoFalta'])
                
                # Ponto T2F
                idx = df_metrics[f'IT2F_Pico_Fase{fase}'].idxmax()
                cols[2].metric(f"Máx I_T2F", f"{df_metrics.loc[idx, f'IT2F_Pico_Fase{fase}']:.1f} A", df_metrics.loc[idx, 'CasoFalta'])
                
                idx = df_metrics[f'VT2F_Pico_Fase{fase}'].idxmin()
                cols[3].metric(f"Mín V_T2F", f"{df_metrics.loc[idx, f'VT2F_Pico_Fase{fase}']:.1f} V", df_metrics.loc[idx, 'CasoFalta'])
                st.markdown("---")

    # --- Abas Pontos (Detalhado) ---
    for i, p in enumerate(PONTOS_BASE):
        with tabs[i+3]:
            clean_p = p.replace('_', '')
            st.markdown(f"### Análise Detalhada: {p}")
            
            # Cria 3 colunas para mostrar A, B, C lado a lado
            cols = st.columns(3)
            for idx_fase, fase in enumerate(FASES):
                with cols[idx_fase]:
                    st.markdown(f"**Fase {fase}**")
                    st.plotly_chart(line_figs[f'I{clean_p}_T_{fase}'], use_container_width=True)
                    st.plotly_chart(line_figs[f'I{clean_p}_F_{fase}'], use_container_width=True)
                    st.plotly_chart(line_figs[f'V{clean_p}_T_{fase}'], use_container_width=True)
                    st.plotly_chart(line_figs[f'V{clean_p}_F_{fase}'], use_container_width=True)

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
                zf.writestr("metricas_completas.csv", df_metrics.to_csv(index=False).encode('utf-8'))
            st.sidebar.download_button("Baixar ZIP", zip_buffer.getvalue(), "resultados.zip", "application/zip")
else:
    st.info("Faça upload dos arquivos .mat.")