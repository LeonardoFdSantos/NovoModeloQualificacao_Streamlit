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
F_MAX_ANALISE = 2000 
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
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=40, b=80),
        hovermode="x unified"
    )
    return fig

# ===================== Função de Processamento =====================
@st.cache_data
def processar_arquivos(uploaded_files):
    colors = qualitative.Plotly
    metrics_data = {'Simulacao': [], 'CasoFalta': [], 'Local_m1': [], 'Cor': []}
    
    for p in PONTOS_BASE:
        for var in ['I', 'V']:
            for fase in FASES:
                for m in METRICAS_TYPES:
                    metrics_data[f'{var}{p}_{m}_Fase{fase}'] = []

    spectrum_data = [] 
    line_figs = {}

    # --- 1. Inicializa Figuras de Linha ---
    for p in PONTOS_BASE:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            key_root = f'{var}{p.replace("_", "")}'
            
            for fase in FASES:
                fig_t_key = f'{key_root}_T_{fase}'
                line_figs[fig_t_key] = go.Figure(layout=go.Layout(title=f'Tempo: {var}_{p} (Fase {fase})', xaxis_title='Tempo (s)', yaxis_title='Amp.'))
                aplicar_estilo_grafico(line_figs[fig_t_key])

                fig_f_key = f'{key_root}_F_{fase}'
                line_figs[fig_f_key] = go.Figure(layout=go.Layout(title=f'FFT: {var}_{p} (Fase {fase})', xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)', xaxis=dict(range=[0, F_MAX_ANALISE/2], autorange=False)))
                aplicar_estilo_grafico(line_figs[fig_f_key])
                
                for h in HARMONICOS_IMPARES:
                    line_figs[fig_f_key].add_vline(x=h*F_FUNDAMENTAL, line_width=0.5, line_dash="dot", line_color="rgba(128, 128, 128, 0.3)")
                    if h <= 9: line_figs[f'{fig_f_key}'].add_annotation(x=h*F_FUNDAMENTAL, y=1, yref="paper", text=f"H{h}", showarrow=False, font_size=8, yshift=10)

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
            
            if 'ts' not in data: raise ValueError("Estrutura inválida (faltam 'ts')")
            struct_ts = data['ts'][0, 0]
            
            for p in PONTOS_BASE:
                for var in ['I', 'V']:
                    vn = f"{var}_{p}"
                    clean_vn_fig = vn.replace('_', '')
                    
                    field_ts = f'ts_{vn}'
                    fig_key_root = clean_vn_fig 

                    try:
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y_all = ts_data_struct['Data']
                        
                        try:
                           struct_fft = data['fft_data'][0, 0]
                           f = struct_fft[f'f_{vn}'].flatten()
                           P1_all = struct_fft[f'P1_{vn}']
                        except:
                           f = np.array([]); P1_all = np.array([])
                        
                        if y_all.shape[1] == 1: y_all = np.tile(y_all, (1, 3))
                        if P1_all.shape[1] == 1: P1_all = np.tile(P1_all, (1, 3))
                        
                        for idx, fase in enumerate(FASES):
                            y_fase = y_all[:, idx]
                            P1_fase = P1_all[:, idx]
                            
                            line_figs[f'{clean_vn_fig}_T_{fase}'].add_trace(go.Scatter(x=t, y=y_fase, name=plotTitle, line=dict(color=currentColor), showlegend=(idx==0)))
                            if f.size > 0:
                                line_figs[f'{clean_vn_fig}_F_{fase}'].add_trace(go.Scatter(x=f, y=20*np.log10(P1_fase+1e-9), name=plotTitle, line=dict(color=currentColor), showlegend=(idx==0)))

                            pico = np.max(np.abs(y_fase)) if y_fase.size > 0 else np.nan
                            thd = calculate_thd(f, P1_fase, F_FUNDAMENTAL, F_MAX_ANALISE) if f.size > 0 else np.nan
                            h3 = get_harmonic_amplitude(f, P1_fase, 3) if f.size > 0 else np.nan
                            h5 = get_harmonic_amplitude(f, P1_fase, 5) if f.size > 0 else np.nan
                            h7 = get_harmonic_amplitude(f, P1_fase, 7) if f.size > 0 else np.nan
                            
                            metrics_data[f'{var}{p}_Pico_Fase{fase}'].append(pico)
                            metrics_data[f'{var}{p}_THD_Fase{fase}'].append(thd)
                            metrics_data[f'{var}{p}_H3_Fase{fase}'].append(h3)
                            metrics_data[f'{var}{p}_H5_Fase{fase}'].append(h5)
                            metrics_data[f'{var}{p}_H7_Fase{fase}'].append(h7)

                        if p == 'T2F' and var == 'I' and f.size > 0:
                            amps_A = [get_amplitude_at_order(f, P1_all[:,0], h) for h in [1] + HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': f"{plotTitle} (Fase A)", 'Amps': amps_A, 'Cor': currentColor})

                    except Exception as e:
                        for fase in FASES:
                            for m in METRICAS_TYPES:
                                metrics_data[f'{var}{p}_{m}_Fase{fase}'].append(np.nan)
        except Exception as e:
            st.error(f"Erro ao ler arquivo {matFile}: {e}")
            continue

    # --- 3. Gráficos de Barra ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)
    
    if not df.empty:
        def create_bar(df_in, y_col, title, y_unit):
            # FIG_CORREÇÃO: Trocar 'CasoFalta' para 'Local_m1' para varredura m1? Não.
            # O eixo X DEVE ser 'CasoFalta' para fins de comparação.
            fig = go.Figure(go.Bar(x=df_in['CasoFalta'], y=df_in[y_col], marker_color=df_in['Cor'], name=title))
            fig.update_layout(title=title, yaxis_title=y_unit, xaxis_tickangle=-45, legend=dict(orientation='h', y=-0.3, x=0.5, xanchor="center"), margin=dict(l=40, r=20, t=40, b=80))
            return fig

        for p in PONTOS_BASE:
            for var in ['I', 'V']:
                unit = 'A' if var == 'I' else 'V'
                for fase in FASES:
                    for m in METRICAS_TYPES:
                        col = f'{var}{p}_{m}_Fase{fase}'
                        if col in df.columns:
                            m_label = "Pico" if m == "Pico" else "THD" if m == "THD" else f"H{m[1]}"
                            bar_figs[f'Bar_{col}'] = create_bar(df, col, f'{m_label} {var}{p} (Fase {fase})', unit if m != 'THD' else '%')

        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title='Espectro Harmônico (I_T2F, Fase A)', yaxis_title='Amplitude (A)', barmode='group')
            bar_figs['Spectrum_Full'] = aplicar_estilo_grafico(sp_fig)

    return line_figs, bar_figs, df

# ===================== Interface =====================
uploaded_files = st.file_uploader("Selecione arquivos .mat", accept_multiple_files=True, type=['.mat'])

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs, df_metrics = processar_arquivos(tuple(uploaded_files))
    st.success("Concluído!")
    
    # Função para Extremos (usada na Aba 0)
    def show_extreme_metric_card(col_obj, df, col, title, find_max=True):
        if col not in df.columns or df[col].isnull().all(): return
        try:
            idx = df[col].idxmax() if find_max else df[col].idxmin()
            label = f"Pior Caso ({'Máx' if find_max else 'Mín'}): {title}"
            val = df.loc[idx, col]
            case = df.loc[idx, 'CasoFalta']
            col_obj.metric(label=label, value=f"{val:.2f}", delta=f"Caso: {case}", delta_color="off")
        except: pass

    # --- Definição das Abas ---
    tab_names = ["🏆 Extremos", "📉 Localização (m1)", "📊 Métricas", "🌊 Espectro"] + [f"📍 {p}" for p in PONTOS_BASE]
    tabs = st.tabs(tab_names)

    # --- Aba 0: Extremos ---
    with tabs[0]:
        st.header("Análise de Casos Extremos (Visão Geral)")
        if not df_metrics.empty:
            for fase in FASES:
                st.subheader(f"Fase {fase}")
                cols = st.columns(4)
                
                # I800 Pico Max
                show_extreme_metric_card(cols[0], df_metrics, f'I800_Pico_Fase{fase}', f"Pico Corrente I_800 (A)", True)
                # I800 THD Max
                show_extreme_metric_card(cols[1], df_metrics, f'I800_THD_Fase{fase}', f"THD Corrente I_800 (%)", True)
                # VT2F Pico Min
                show_extreme_metric_card(cols[2], df_metrics, f'VT2F_Pico_Fase{fase}', f"Tensão Mínima V_T2F (V)", False)
                # I T2F H3 Max
                show_extreme_metric_card(cols[3], df_metrics, f'IT2F_H3_Fase{fase}', f"H3 Corrente I_T2F (A)", True)
                st.markdown("---")

    # --- Aba 1: Varredura m1 ---
    with tabs[1]:
        st.header("Análise de Localização de Falta (Varredura m1)")
        df_m1 = df_metrics[df_metrics['Local_m1'] > 0].copy()
        
        if not df_m1.empty:
            c1, c2, c3 = st.columns(3)
            sim_choice = c1.selectbox("Simulação:", df_m1['Simulacao'].unique())
            fase_choice = c2.selectbox("Fase:", FASES)
            
            metric_cols = [col for col in df_m1.columns if ('Pico' in col or 'THD' in col) and f'Fase{fase_choice}' in col]
            metric_choice = c3.selectbox("Métrica:", metric_cols)
            
            df_plot_sim = df_m1[df_m1['Simulacao'] == sim_choice]
            
            fig_m1 = go.Figure()
            for case in df_plot_sim['CasoFalta'].unique():
                df_case = df_plot_sim[df_plot_sim['CasoFalta'] == case].sort_values('Local_m1')
                if not df_case.empty:
                    fig_m1.add_trace(go.Scatter(x=df_case['Local_m1'], y=df_case[metric_choice], mode='lines+markers', name=case))
            
            fig_m1.update_layout(xaxis_title="Localização (m1)", yaxis_title=metric_choice)
            st.plotly_chart(fig_m1, use_container_width=True)
        else:
            st.info("Nenhum caso com variação de m1 encontrado.")

    # --- Aba 2: Métricas (Barras Trifásicas) ---
    with tabs[2]:
        fase_bar_select = st.radio("Visualizar Fase:", FASES, horizontal=True, key="bar_fase")
        st.markdown(f"### Comparação Quantitativa (Fase {fase_bar_select})")
        
        for p in PONTOS_BASE:
            with st.expander(f"Dados do Ponto {p}", expanded=(p=='800')):
                cols = st.columns(3)
                base = f'I{p.replace("_", "")}_Pico_Fase{fase_bar_select}'
                if f'Bar_{base}' in bar_figs: cols[0].plotly_chart(bar_figs[f'Bar_{base}'], use_container_width=True)
                base = f'I{p.replace("_", "")}_THD_Fase{fase_bar_select}'
                if f'Bar_{base}' in bar_figs: cols[1].plotly_chart(bar_figs[f'Bar_{base}'], use_container_width=True)
                base = f'I{p.replace("_", "")}_H3_Fase{fase_bar_select}'
                if f'Bar_{base}' in bar_figs: cols[2].plotly_chart(bar_figs[f'Bar_{base}'], use_container_width=True)
                
                cols_v = st.columns(3)
                base = f'V{p.replace("_", "")}_Pico_Fase{fase_bar_select}'
                if f'Bar_{base}' in bar_figs: cols_v[0].plotly_chart(bar_figs[f'Bar_{base}'], use_container_width=True)
                base = f'V{p.replace("_", "")}_THD_Fase{fase_bar_select}'
                if f'Bar_{base}' in bar_figs: cols_v[1].plotly_chart(bar_figs[f'Bar_{base}'], use_container_width=True)
                base = f'V{p.replace("_", "")}_H3_Fase{fase_bar_select}'
                if f'Bar_{base}' in bar_figs: cols_v[2].plotly_chart(bar_figs[f'Bar_{base}'], use_container_width=True)

    # --- Aba 3: Espectro ---
    with tabs[3]:
        if 'Spectrum_Full' in bar_figs: st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

    # --- Abas 4+: Pontos (Detalhes da Onda e FFT) ---
    for i, p in enumerate(PONTOS_BASE):
        with tabs[i+4]:
            clean_p = p.replace('_', '')
            st.markdown(f"### Análise Detalhada: Ponto {p}")
            
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