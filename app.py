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
st.set_page_config(layout="wide", page_title="Analisador de Harmônicas T2F (10kHz)")
st.title("Analisador Avançado de Qualidade de Energia (5 Pontos)")

# ===================== Constantes =====================
F_FUNDAMENTAL = 60  # Hz
F_MAX_ANALISE = 10000 # 10 kHz
HARMONICOS_IMPARES = [h for h in range(3, int(F_MAX_ANALISE / F_FUNDAMENTAL) + 1, 2)]
PONTOS = ['800', 'T2F', '818', '820', '822']

# ===================== Funções Auxiliares =====================
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    """Encontra a amplitude de uma frequência específica."""
    target_freq = order * fund_freq
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]

def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    """Calcula THD."""
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
    metrics_data = {'Caso': [], 'Cor': []}
    
    metricas_types = ['Pico', 'THD', 'H3', 'H5', 'H7']
    for p in PONTOS:
        for var in ['I', 'V']:
            for m in metricas_types:
                metrics_data[f'{var}{p}_{m}'] = []

    spectrum_data = [] 
    line_figs = {}

    for p in PONTOS:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            key_root = f'{var}{p}' 
            line_figs[f'{key_root}_T'] = go.Figure(layout=go.Layout(
                title=f'Sinal no Tempo: {label} ({var}_{p}) - Fase A', 
                xaxis_title='Tempo (s)', yaxis_title='Amplitude'))
            line_figs[f'{key_root}_F'] = go.Figure(layout=go.Layout(
                title=f'Espectro FFT: {label} ({var}_{p}) - Fase A', 
                xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)', 
                xaxis=dict(range=[0, F_MAX_ANALISE], autorange=False)))
            
            for h in HARMONICOS_IMPARES:
                line_figs[f'{key_root}_F'].add_vline(x=h*F_FUNDAMENTAL, line_width=0.5, line_dash="dot", line_color="rgba(128, 128, 128, 0.3)")
                if h <= 15: 
                    line_figs[f'{key_root}_F'].add_annotation(x=h*F_FUNDAMENTAL, y=1, yref="paper", text=f"H{h}", showarrow=False, font=dict(size=8, color="gray"), yshift=10)

    # --- Loop Principal de Arquivos ---
    for i, file in enumerate(uploaded_files):
        matFile = file.name
        currentColor = colors[i % len(colors)]
        try:
            data = sio.loadmat(io.BytesIO(file.read()))
            plotTitle = matFile.replace('.mat', '').replace('__', ' - ').replace('_', ' ')
            metrics_data['Caso'].append(plotTitle)
            metrics_data['Cor'].append(currentColor)

            try:
                struct_ts = data['ts'][0, 0]
                struct_fft = data['fft_data'][0, 0]
            except KeyError:
                st.error(f"Estrutura inválida no arquivo {matFile}.")
                continue

            for p in PONTOS:
                for var in ['I', 'V']:
                    vn = f"{var}_{p}" 
                    clean_vn = vn.replace('_', '') 
                    field_ts = f'ts_{vn}'   
                    field_f = f'f_{vn}'     
                    field_p1 = f'P1_{vn}'   
                    fig_key_root = clean_vn 

                    try:
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y = ts_data_struct['Data'][:, 0] 
                        f = struct_fft[field_f].flatten()
                        P1 = struct_fft[field_p1][:, 0] 

                        line_figs[f'{fig_key_root}_T'].add_trace(go.Scatter(x=t, y=y, name=plotTitle, line=dict(color=currentColor)))
                        line_figs[f'{fig_key_root}_F'].add_trace(go.Scatter(x=f, y=20*np.log10(P1+1e-9), name=plotTitle, line=dict(color=currentColor)))

                        pico = np.max(np.abs(y))
                        thd = calculate_thd(f, P1, fund_freq=F_FUNDAMENTAL, max_freq=F_MAX_ANALISE)
                        h3 = get_harmonic_amplitude(f, P1, 3)
                        h5 = get_harmonic_amplitude(f, P1, 5)
                        h7 = get_harmonic_amplitude(f, P1, 7)

                        metrics_data[f'{fig_key_root}_Pico'].append(pico)
                        metrics_data[f'{fig_key_root}_THD'].append(thd)
                        metrics_data[f'{fig_key_root}_H3'].append(h3)
                        metrics_data[f'{fig_key_root}_H5'].append(h5)
                        metrics_data[f'{fig_key_root}_H7'].append(h7)

                        if p == 'T2F' and var == 'I':
                            amps = [get_harmonic_amplitude(f, P1, h) for h in [1] + HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': plotTitle, 'Amps': amps, 'Cor': currentColor})
                    except ValueError:
                        for m in metricas_types: metrics_data[f'{fig_key_root}_{m}'].append(0)
        except Exception as e:
            st.error(f"Erro fatal ao ler {matFile}: {e}")
            continue

    # --- Criação dos Gráficos de Barra ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)
    
    if not df.empty:
        def create_bar(y_col, title_text, y_label):
            fig = go.Figure(go.Bar(x=df['Caso'], y=df[y_col], marker_color=df['Cor'], name=title_text))
            fig.update_layout(title=title_text, yaxis_title=y_label, xaxis_tickangle=-45)
            return fig

        for p in PONTOS:
            for var in ['I', 'V']:
                base = f'{var}{p}'
                for m in metricas_types:
                    metric_key = f'{base}_{m}' # Ex: I800_Pico
                    # Títulos amigáveis
                    m_label = "Pico" if m == "Pico" else "THD" if m == "THD" else f"{m[1]}ª Harmônica"
                    unit = "A" if var == "I" else "V"
                    if m == "THD": unit = "%"
                    bar_figs[f'Bar_{metric_key}'] = create_bar(metric_key, f'{m_label} {var} ({p})', unit)

        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title=f'Espectro Harmônico (I_T2F) - H1 até ~H{HARMONICOS_IMPARES[-1]}', 
                                   yaxis_title='Amplitude (A)', barmode='group', xaxis_tickangle=-90)
            bar_figs['Spectrum_Full'] = sp_fig

    # RETORNO ATUALIZADO: Retorna também o DataFrame
    return line_figs, bar_figs, df

# ===================== Interface =====================
uploaded_files = st.file_uploader(
    "Selecione arquivos .mat",
    accept_multiple_files=True, type=['.mat']
)

if uploaded_files:
    with st.spinner('Processando...'):
        # RECEBIMENTO ATUALIZADO: Recebe o df_metrics
        line_figs, bar_figs, df_metrics = processar_arquivos(tuple(uploaded_files))

    st.success("Concluído!")
    
    # --- (NOVO) Função para a Aba de Extremos ---
    def show_extreme_metric(df, col, title, find_max=True):
        """Função helper para encontrar e exibir o caso extremo com st.metric."""
        if col not in df.columns:
            st.warning(f"Métrica {col} não encontrada.")
            return
        
        try:
            if find_max:
                # Pior Caso = Máximo (Corrente, THD)
                extreme_idx = df[col].idxmax()
                extreme_val = df.loc[extreme_idx, col]
                extreme_case = df.loc[extreme_idx, 'Caso']
                label = f"Pior Caso (Máx): {title}"
            else:
                # Pior Caso = Mínimo (Tensão)
                extreme_idx = df[col].idxmin()
                extreme_val = df.loc[extreme_idx, col]
                extreme_case = df.loc[extreme_idx, 'Caso']
                label = f"Pior Caso (Mín): {title}"

            st.metric(label=label, value=f"{extreme_val:.2f}")
            st.caption(f"Caso: {extreme_case}")
        
        except Exception as e:
            st.error(f"Erro ao analisar {col}: {e}")

    # --- (NOVO) Definição das Abas ---
    tab_names = ["🏆 Análise de Extremos", "📊 Métricas (Barras)", "🌊 Espectro (I_T2F)"] + [f"📍 {p}" for p in PONTOS]
    tabs = st.tabs(tab_names)

    # --- (NOVA) Aba 0: Análise de Extremos ---
    with tabs[0]:
        st.header("Análise de Casos Extremos (Mínimos e Máximos)")
        st.info("Esta aba identifica os piores casos (maiores correntes, maior distorção, maiores afundamentos) dentre todos os arquivos carregados.")
        
        if df_metrics.empty:
            st.error("Nenhum dado válido foi processado.")
        else:
            st.subheader("⚡ Piores Casos de CORRENTE (Pico e THD)")
            col1, col2 = st.columns(2)
            with col1:
                show_extreme_metric(df_metrics, 'I800_Pico', "Pico de Corrente (I_800)", find_max=True)
            with col2:
                show_extreme_metric(df_metrics, 'I800_THD', "THD de Corrente (I_800)", find_max=True)
            
            with col1:
                show_extreme_metric(df_metrics, 'IT2F_Pico', "Pico de Corrente (I_T2F)", find_max=True)
            with col2:
                show_extreme_metric(df_metrics, 'IT2F_THD', "THD de Corrente (I_T2F)", find_max=True)
                
            st.markdown("---")
            st.subheader("📉 Piores Casos de TENSÃO (Afundamento)")
            col3, col4 = st.columns(2)
            with col3:
                show_extreme_metric(df_metrics, 'V800_Pico', "Tensão Mínima (V_800)", find_max=False)
            with col4:
                show_extreme_metric(df_metrics, 'VT2F_Pico', "Tensão Mínima (V_T2F)", find_max=False)
                
            st.markdown("---")
            st.subheader("🕊️ Melhores Casos (Operação Normal / Base)")
            try:
                # Filtra o DataFrame para encontrar o caso "Sem Falta"
                sem_falta_rows = df_metrics[df_metrics['Caso'].str.contains("Sem Falta")]
                if not sem_falta_rows.empty:
                    st.dataframe(sem_falta_rows)
                else:
                    st.warning("Nenhum caso 'Sem Falta' foi encontrado para usar como base.")
            except Exception as e:
                st.error(f"Não foi possível encontrar o caso 'Sem Falta': {e}")


    # Aba 1: Métricas
    with tabs[1]:
        st.markdown("### Comparação de Harmônicas e THD")
        for p in PONTOS:
            with st.expander(f"Dados do Ponto {p}", expanded=(p=='T2F')):
                st.markdown(f"#### Corrente ({p})")
                c1, c2, c3 = st.columns(3)
                c1.plotly_chart(bar_figs[f'Bar_I{p}_Pico'], use_container_width=True)
                c2.plotly_chart(bar_figs[f'Bar_I{p}_THD'], use_container_width=True)
                c3.plotly_chart(bar_figs[f'Bar_I{p}_H3'], use_container_width=True)
                
                st.markdown(f"#### Tensão ({p})")
                c4, c5, c6 = st.columns(3)
                c4.plotly_chart(bar_figs[f'Bar_V{p}_Pico'], use_container_width=True)
                c5.plotly_chart(bar_figs[f'Bar_V{p}_THD'], use_container_width=True)
                c6.plotly_chart(bar_figs[f'Bar_V{p}_H3'], use_container_width=True)

    # Aba 2: Espectro
    with tabs[2]:
        if 'Spectrum_Full' in bar_figs:
            st.markdown("#### Varredura Espectral Completa (Ímpares até 10 kHz)")
            st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)
        else:
            st.info("Nenhum dado de espectro disponível.")

    # Abas de Pontos (3 a 7)
    for i, p in enumerate(PONTOS):
        with tabs[i+3]: # Inicia no índice 3
            st.markdown(f"### Análise Detalhada: Ponto {p}")
            c1, c2 = st.columns(2)
            c1.plotly_chart(line_figs[f'I{p}_T'], use_container_width=True)
            c2.plotly_chart(line_figs[f'I{p}_F'], use_container_width=True)
            c3, c4 = st.columns(2)
            c3.plotly_chart(line_figs[f'V{p}_T'], use_container_width=True)
            c4.plotly_chart(line_figs[f'V{p}_F'], use_container_width=True)

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
            st.sidebar.download_button("Baixar ZIP", zip_buffer.getvalue(), "resultados.zip", "application/zip")
else:
    st.info("Faça upload dos arquivos .mat.")