import streamlit as st
import os
import io
import zipfile
import plotly.graph_objects as go
from plotly.colors import qualitative
import scipy.io as sio
import numpy as np
import pandas as pd
from scipy.signal.windows import flattop

# ===================== Configuração da Página =====================
st.set_page_config(layout="wide", page_title="Analisador Trifásico T2F")
st.title("Analisador de Faltas - Varredura Trifásica")

# ===================== Constantes =====================
F_FUNDAMENTAL = 60  # Hz
F_MAX_ANALISE = 2000 
HARMONICOS_IMPARES = [h for h in range(3, int(F_MAX_ANALISE / F_FUNDAMENTAL) + 1, 2) if h <= 15]

PONTOS_BASE = ['800', 'T2F', '818_1', '818_2', '820', '822']
METRICAS_CALC = ['Pico', 'THD', 'H3', 'H5', 'H7']
FASES = ['A', 'B', 'C']

# ===================== Funções de Processamento FFT (Em Python) =====================
def get_amplitude_at_order(freq_array, amp_array, order, fund_freq=60):
    """Encontra a amplitude de uma frequência específica (FFT bin)."""
    target_freq = order * fund_freq
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]

def calculate_fft_and_harmonics(time_vec, data_vec, fund_freq=60):
    """Executa a FFT completa (com Janelamento) e extrai métricas."""
    
    if data_vec.size < 4: return None, None, {}

    Ts = np.mean(np.diff(time_vec))
    if Ts == 0: return None, None, {}
    
    Fs = 1.0 / Ts
    L = len(data_vec)
    
    # 1. Janela Flat Top (para precisão de amplitude)
    win = flattop(L, sym=False)
    y_janelado = data_vec * win
    
    # 2. FFT
    Y = np.fft.fft(y_janelado)
    
    # 3. Correção de Amplitude (ganho coerente)
    coherent_gain = np.sum(win) / L
    P2 = np.abs(Y / (L * coherent_gain))
    
    # Espectro de Lado Único (SSAS)
    P1 = P2[:L//2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    
    # Vetor de Frequência
    f = np.linspace(0, Fs/2, L//2 + 1)

    # 4. Cálculo de Harmônicas e THD
    h_metrics = {}
    
    def get_amp(order):
        return get_amplitude_at_order(f, P1, order, fund_freq)
        
    p1_array = P1 
    
    # Cálculo THD (apenas H2 até H50 para THD padrão)
    sum_squares = 0
    for order in range(2, 51): # THD calculado até a 50ª ordem
        sum_squares += get_amp(order) ** 2
        
    amp_h1 = get_amp(1)
    h_metrics['THD'] = (np.sqrt(sum_squares) / amp_h1) * 100 if amp_h1 > 1e-9 else 0
    
    # Harmônicas específicas
    h_metrics['H3'] = get_amp(3)
    h_metrics['H5'] = get_amp(5)
    h_metrics['H7'] = get_amp(7)

    return f, P1, h_metrics

# ===================== Função de Processamento Principal =====================
@st.cache_data
def processar_arquivos(uploaded_files):
    colors = qualitative.Plotly
    metrics_data = {'Simulacao': [], 'CasoFalta': [], 'Local_m1': [], 'Cor': []}
    
    # Inicializa colunas do DataFrame
    for p in PONTOS_BASE:
        for var in ['I', 'V']:
            for fase in FASES:
                for m in METRICAS_CALC:
                    metrics_data[f'{var}{p}_{m}_Fase{fase}'] = []

    spectrum_data = [] 
    line_figs = {}

    # Inicializa Figuras de Linha
    for p in PONTOS_BASE:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            key_root = f'{var}{p.replace("_", "")}'
            
            for fase in FASES:
                fig_key = f'{key_root}_T_{fase}'
                line_figs[fig_key] = go.Figure(layout=go.Layout(title=f'Tempo: {var}_{p} (Fase {fase})', xaxis_title='s', yaxis_title='Amp.'))
                aplicar_estilo_grafico(line_figs[fig_key])

                fig_key = f'{key_root}_F_{fase}'
                line_figs[fig_key] = go.Figure(layout=go.Layout(title=f'FFT: {var}_{p} (Fase {fase})', xaxis_title='Hz', yaxis_title='dB', xaxis=dict(range=[0, F_MAX_ANALISE/2], autorange=False)))
                aplicar_estilo_grafico(line_figs[fig_key])
                
                for h in HARMONICOS_IMPARES:
                    line_figs[fig_key].add_vline(x=h*F_FUNDAMENTAL, line_width=0.5, line_dash="dot", line_color="gray", opacity=0.3)
                    if h <= 9: line_figs[fig_key].add_annotation(x=h*F_FUNDAMENTAL, y=1, yref="paper", text=f"H{h}", showarrow=False, font_size=8, yshift=10)

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
            
            try: m1_loc = data['m1_location'][0, 0]; 
            except: m1_loc = 0;
            metrics_data['Local_m1'].append(m1_loc)
            
            struct_ts = data['ts'][0, 0]
            
            # --- Processamento por Ponto ---
            for p in PONTOS_BASE:
                for var in ['I', 'V']:
                    vn = f"{var}_{p}"
                    clean_vn_fig = vn.replace('_', '')
                    
                    field_ts = f'ts_{vn}'
                    fig_key_root = clean_vn_fig 

                    try:
                        # Extração dos dados brutos (Time/Data)
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y_all = ts_data_struct['Data']
                        
                        if y_all.shape[1] == 1: y_all = np.tile(y_all, (1, 3)) # Garante 3 fases
                        
                        # Loop por Fase
                        for idx, fase in enumerate(FASES):
                            y_fase = y_all[:, idx]
                            
                            # 1. FFT, H3, THD são calculados AQUI
                            f, P1_f, h_metrics = calculate_fft_and_harmonics(t, y_fase, F_FUNDAMENTAL)
                            
                            # Plotagem
                            line_figs[f'{fig_key_root}_T_{fase}'].add_trace(go.Scatter(x=t, y=y_fase, name=plotTitle, line=dict(color=currentColor), showlegend=(idx==0)))
                            if f is not None:
                                line_figs[f'{fig_key_root}_F_{fase}'].add_trace(go.Scatter(x=f, y=20*np.log10(P1_f+1e-9), name=plotTitle, line=dict(color=currentColor), showlegend=(idx==0)))

                            # Métricas
                            metrics_data[f'{var}{p}_Pico_Fase{fase}'].append(np.max(np.abs(y_fase)))
                            metrics_data[f'{var}{p}_THD_Fase{fase}'].append(h_metrics['THD'])
                            metrics_data[f'{var}{p}_H3_Fase{fase}'].append(h_metrics['H3'])
                            metrics_data[f'{var}{p}_H5_Fase{fase}'].append(h_metrics['H5'])
                            metrics_data[f'{var}{p}_H7_Fase{fase}'].append(h_metrics['H7'])

                        # Espectro (T2F, Fase A)
                        if p == 'T2F' and var == 'I' and f is not None:
                            amps_A = [get_amplitude_at_order(f, P1_f, h) for h in [1] + HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': f"{plotTitle} (Fase A)", 'Amps': amps_A, 'Cor': currentColor})

                    except Exception:
                        # Se falhar, preenche com NaN
                        for fase in FASES:
                            for m in METRICAS_TYPES:
                                metrics_data[f'{var}{p}_{m}_Fase{fase}'].append(np.nan)
        except Exception as e:
            st.error(f"Erro ao ler {matFile}: {e}")
            continue

    # --- 3. Gráficos de Barra ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)
    
    # ... (Bar chart logic remains the same, using the new metrics) ...
    if not df.empty:
        def create_bar(df_in, y_col, title, y_unit):
            fig = go.Figure(go.Bar(x=df_in['CasoFalta'], y=df_in[y_col], marker_color=df_in['Cor'], name=title))
            fig.update_layout(title=title, yaxis_title=y_unit, xaxis_tickangle=-45, legend=dict(orientation='h', y=-0.3, x=0.5, xanchor="center"), margin=dict(l=40, r=20, t=40, b=80))
            return fig

        for p in PONTOS_BASE:
            for var in ['I', 'V']:
                unit = 'A' if var == 'I' else 'V'
                for fase in FASES:
                    for m in METRICAS_TYPES:
                        col_name = f'{var}{p}_{m}_Fase{fase}'
                        if col_name in df.columns:
                            m_label = "Pico" if m == "Pico" else "THD" if m == "THD" else f"H{m[1]}"
                            bar_figs[f'Bar_{col_name}'] = create_bar(df, col_name, f'{m_label} {var}{p} (Fase {fase})', unit if m != 'THD' else '%')

        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title='Espectro Harmônico (I_T2F, Fase A)', yaxis_title='Amplitude (A)', barmode='group')
            bar_figs['Spectrum_Full'] = sp_fig

    return line_figs, bar_figs, df

# ===================== Interface =====================
uploaded_files = st.file_uploader("Selecione arquivos .mat", accept_multiple_files=True, type=['.mat'])

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs, df_metrics = processar_arquivos(tuple(uploaded_files))
    st.success("Concluído!")
    
    # --- Definição das Abas ---
    tab_names = ["🏆 Extremos", "📉 Localização (m1)", "📊 Métricas"] + [f"📍 {p}" for p in PONTOS_BASE]
    tabs = st.tabs(tab_names)

    # --- Aba 0: Extremos ---
    with tabs[0]:
        st.header("Análise de Casos Extremos (Todas as Fases)")
        if not df_metrics.empty:
            for fase in FASES:
                st.subheader(f"Fase {fase}")
                cols = st.columns(4)
                
                # I800 Pico Max
                show_extreme_metric_card(cols[0], df_metrics, f'I800_Pico_Fase{fase}', "Max I_800 (A)", find_max=True)
                # I800 THD Max
                show_extreme_metric_card(cols[1], df_metrics, f'I800_THD_Fase{fase}', "Max THD I_800 (%)", find_max=True)
                # VT2F Pico Min
                show_extreme_metric_card(cols[2], df_metrics, f'VT2F_Pico_Fase{fase}', "Min V_T2F (V)", find_max=False)
                # IT2F H3 Max
                show_extreme_metric_card(cols[3], df_metrics, f'IT2F_H3_Fase{fase}', "Max H3 I_T2F (A)", find_max=True)
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

    # --- Aba 2: Métricas (Barras) ---
    with tabs[2]:
        for p in PONTOS_BASE:
            with st.expander(f"Dados do Ponto {p}", expanded=(p=='800')):
                for var in ['I', 'V']:
                    st.markdown(f"#### {var} {p}")
                    cols = st.columns(3)
                    for i, fase in enumerate(FASES):
                        col_key = f'Bar_{var}{p.replace("_", "")}_{"Pico"}_Fase{fase}'
                        if col_key in bar_figs: cols[i].plotly_chart(bar_figs[col_key], use_container_width=True)
                    
                    st.markdown("Harmônicas:")
                    c1, c2, c3 = st.columns(3)
                    if f'Bar_{var}{p.replace("_", "")}_THD_FaseA' in bar_figs: c1.plotly_chart(bar_figs[f'Bar_{var}{p.replace("_", "")}_THD_FaseA'], use_container_width=True)
                    if f'Bar_{var}{p.replace("_", "")}_H3_FaseA' in bar_figs: c2.plotly_chart(bar_figs[f'Bar_{var}{p.replace("_", "")}_H3_FaseA'], use_container_width=True)
                    if f'Bar_{var}{p.replace("_", "")}_H5_FaseA' in bar_figs: c3.plotly_chart(bar_figs[f'Bar_{var}{p.replace("_", "")}_H5_FaseA'], use_container_width=True)

    # --- Aba 3: Espectro (I_T2F) ---
    with tabs[3]:
        if 'Spectrum_Full' in bar_figs: st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

    # --- Abas 4+: Pontos (Detalhes da Onda e FFT) ---
    for i, p in enumerate(PONTOS_BASE):
        with tabs[i+4]:
            st.markdown(f"### Análise Detalhada: Ponto {p}")
            st.markdown("---")
            
            # Sub-abas para as fases (A, B, C)
            subtab_A, subtab_B, subtab_C = st.tabs(["Fase A", "Fase B", "Fase C"])
            
            for sub_tab, fase in zip([subtab_A, subtab_B, subtab_C], FASES):
                with sub_tab:
                    key_I = f"I{p.replace('_', '')}"
                    key_V = f"V{p.replace('_', '')}"
                    c1, c2 = st.columns(2)
                    c1.plotly_chart(line_figs[f'{key_I}_T_{fase}'], use_container_width=True)
                    c2.plotly_chart(line_figs[f'{key_I}_F_{fase}'], use_container_width=True)
                    c3, c4 = st.columns(2)
                    c3.plotly_chart(line_figs[f'{key_V}_T_{fase}'], use_container_width=True)
                    c4.plotly_chart(line_figs[f'{key_V}_F_{fase}'], use_container_width=True)

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

# --- Helper para Extremos (Adicionado no final para escopo) ---
def show_extreme_metric_card(col_obj, df, col, title, find_max=True):
    if col not in df.columns or df[col].isnull().all(): return
    try:
        idx = df[col].idxmax() if find_max else df[col].idxmin()
        label = f"Pior Caso ({'Máx' if find_max else 'Mín'}): {title}"
        val = df.loc[idx, col]
        case = df.loc[idx, 'CasoFalta']
        col_obj.metric(label=label, value=f"{val:.2f}", delta=f"Caso: {case}", delta_color="off")
    except: pass