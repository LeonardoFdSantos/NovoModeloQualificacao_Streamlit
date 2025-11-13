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
st.title("Analisador Avançado de Qualidade de Energia (MATLAB/Python)")

# ===================== Constantes e Funções Auxiliares =====================
F_FUNDAMENTAL = 60  # Hz
PLOT_XLIM_FFT = [0, 1000]
HARMONICOS_IMPARES = [3, 5, 7, 9, 11, 13, 15]  # Harmônicas para marcar nos gráficos


# Função para encontrar a amplitude de uma frequência específica nos dados da FFT
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    target_freq = order * fund_freq
    # Encontra o índice mais próximo da frequência alvo
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]


# Função para calcular THD (Total Harmonic Distortion)
def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    # Encontra amplitude fundamental
    amp_h1 = get_harmonic_amplitude(freq_array, amp_array, 1, fund_freq)

    if amp_h1 == 0: return 0

    sum_squares = 0
    # Soma os quadrados das harmônicas (inteiras) até max_freq
    for order in range(2, int(max_freq / fund_freq)):
        amp_hn = get_harmonic_amplitude(freq_array, amp_array, order, fund_freq)
        sum_squares += amp_hn ** 2

    thd = (np.sqrt(sum_squares) / amp_h1) * 100  # Em porcentagem
    return thd


# Mapeamento de nomes para download
FNAMES_MAP = {
    'I800_T': 'Comparacao_I800_Tempo', 'I800_F': 'Comparacao_I800_FFT',
    'V800_T': 'Comparacao_V800_Tempo', 'V800_F': 'Comparacao_V800_FFT',
    'IT2F_T': 'Comparacao_IT2F_Tempo', 'IT2F_F': 'Comparacao_IT2F_FFT',
    'VT2F_T': 'Comparacao_VT2F_Tempo', 'VT2F_F': 'Comparacao_VT2F_FFT',
    'Bar_I_Pico': 'Bar_I800_Pico', 'Bar_I_THD': 'Bar_I800_THD',
    'Bar_I_h3': 'Bar_I800_H3', 'Bar_I_h5': 'Bar_I800_H5',
    'Bar_V_Pico': 'Bar_V800_Pico', 'Bar_V_THD': 'Bar_V800_THD'
}


# ===================== Função de Processamento =====================
@st.cache_data
def processar_arquivos(uploaded_files):
    numFiles = len(uploaded_files)
    colors = qualitative.Plotly

    # Listas para armazenar dados
    legend_entries = []

    # DataFrames para armazenar métricas de TODOS os arquivos
    metrics_data = {
        'Caso': [],
        # Ponto 800 - Corrente
        'I800_Pico': [], 'I800_THD': [], 'I800_H3': [], 'I800_H5': [], 'I800_H7': [],
        # Ponto 800 - Tensão
        'V800_Pico': [], 'V800_THD': [], 'V800_H3': [], 'V800_H5': [], 'V800_H7': [],
        # Cores
        'Cor': []
    }

    # Dados para o gráfico de espectro completo (H1-H15)
    spectrum_data = []

    # --- Inicializa Figuras de Linha ---
    line_figs = {}
    pairs = [('I800', 'Corrente (I_800)'), ('V800', 'Tensão (V_800)'),
             ('IT2F', 'Corrente (I_T2F)'), ('VT2F', 'Tensão (V_T2F)')]

    for key, title_desc in pairs:
        # Tempo
        line_figs[f'{key}_T'] = go.Figure(layout=go.Layout(
            title=f'Sinal no Tempo: {title_desc} - Fase A', xaxis_title='Tempo (s)',
            yaxis_title='Amplitude'))

        # FFT
        line_figs[f'{key}_F'] = go.Figure(layout=go.Layout(
            title=f'Espectro FFT: {title_desc} - Fase A', xaxis_title='Frequência (Hz)',
            yaxis_title='Amplitude (dB)', xaxis_range=PLOT_XLIM_FFT))

        # Adiciona linhas verticais para harmônicas ímpares
        for h in HARMONICOS_IMPARES:
            line_figs[f'{key}_F'].add_vline(x=h * F_FUNDAMENTAL, line_width=1, line_dash="dot",
                                            line_color="gray", opacity=0.5)
            # Adiciona anotação apenas para H3 e H5 para não poluir
            if h in [3, 5]:
                line_figs[f'{key}_F'].add_annotation(x=h * F_FUNDAMENTAL, y=0, text=f"H{h}",
                                                     showarrow=False, yshift=10)

    # --- Loop Principal ---
    for i, file in enumerate(uploaded_files):
        matFile = file.name
        currentColor = colors[i % len(colors)]

        try:
            data = sio.loadmat(io.BytesIO(file.read()))
            plotTitle = matFile.replace('.mat', '').replace('_analise_completa', '').replace('__', ' - ').replace('_',
                                                                                                                  ' ')

            legend_entries.append(plotTitle)
            metrics_data['Caso'].append(plotTitle)
            metrics_data['Cor'].append(currentColor)

            # --- Extração e Plotagem (800 e T2F) ---
            # Helper para processar cada sinal
            def process_signal(prefix_ts, prefix_fft, fig_key_t, fig_key_f, is_current=True):
                # Extrai dados
                t = data[prefix_ts][0, 0]['Time'].flatten()
                y = data[prefix_ts][0, 0]['Data'][:, 0]
                f = data['fft_data'][0, 0][prefix_fft].flatten()
                P1 = data['fft_data'][0, 0][prefix_fft.replace('f_', 'P1_')][:, 0]

                # Plota Linhas
                line_figs[fig_key_t].add_trace(go.Scatter(x=t, y=y, name=plotTitle, line=dict(color=currentColor)))
                line_figs[fig_key_f].add_trace(
                    go.Scatter(x=f, y=20 * np.log10(P1 + 1e-9), name=plotTitle, line=dict(color=currentColor)))

                # Calcula Métricas (Apenas se for Ponto 800 para os gráficos de barra principais)
                if '800' in fig_key_t:
                    pico = np.max(np.abs(y))
                    thd = calculate_thd(f, P1)
                    h3 = get_harmonic_amplitude(f, P1, 3)
                    h5 = get_harmonic_amplitude(f, P1, 5)
                    h7 = get_harmonic_amplitude(f, P1, 7)

                    type_key = 'I800' if is_current else 'V800'
                    metrics_data[f'{type_key}_Pico'].append(pico)
                    metrics_data[f'{type_key}_THD'].append(thd)
                    metrics_data[f'{type_key}_H3'].append(h3)
                    metrics_data[f'{type_key}_H5'].append(h5)
                    metrics_data[f'{type_key}_H7'].append(h7)

                    # Dados para o espectro completo (apenas Corrente I800)
                    if is_current:
                        amps = [get_harmonic_amplitude(f, P1, h) for h in [1] + HARMONICOS_IMPARES]
                        spectrum_data.append({'Caso': plotTitle, 'Amps': amps, 'Cor': currentColor})

            # Processa os 4 sinais
            process_signal('ts_I_800', 'f_I800', 'I800_T', 'I800_F', True)
            process_signal('ts_V_800', 'f_V800', 'V800_T', 'V800_F', False)
            process_signal('ts_I_T2F', 'f_IT2F', 'IT2F_T', 'IT2F_F', True)
            process_signal('ts_V_T2F', 'f_VT2F', 'VT2F_T', 'VT2F_F', False)

        except Exception as e:
            st.error(f"Erro no arquivo '{matFile}': {e}")
            continue

    # --- Criação dos Gráficos de Barra (Bar Figs) ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)

    if not df.empty:
        # Função auxiliar para criar barras
        def create_bar(y_col, title, y_axis):
            fig = go.Figure(go.Bar(x=df['Caso'], y=df[y_col], marker_color=df['Cor'], name=title))
            fig.update_layout(title=title, yaxis_title=y_axis, xaxis_tickangle=-45)
            return fig

        # Métricas de Corrente (I800)
        bar_figs['Bar_I_Pico'] = create_bar('I800_Pico', 'Pico de Corrente (I_800)', 'Corrente (A)')
        bar_figs['Bar_I_THD'] = create_bar('I800_THD', 'THD de Corrente (I_800)', 'THD (%)')
        bar_figs['Bar_I_h3'] = create_bar('I800_H3', '3ª Harmônica (180 Hz)', 'Amplitude (A)')
        bar_figs['Bar_I_h5'] = create_bar('I800_H5', '5ª Harmônica (300 Hz)', 'Amplitude (A)')
        bar_figs['Bar_I_h7'] = create_bar('I800_H7', '7ª Harmônica (420 Hz)', 'Amplitude (A)')

        # Métricas de Tensão (V800)
        bar_figs['Bar_V_Pico'] = create_bar('V800_Pico', 'Pico de Tensão (V_800)', 'Tensão (V)')
        bar_figs['Bar_V_THD'] = create_bar('V800_THD', 'THD de Tensão (V_800)', 'THD (%)')
        bar_figs['Bar_V_h3'] = create_bar('V800_H3', '3ª Harmônica Tensão', 'Amplitude (V)')

        # --- Gráfico de Espectro Completo (Grouped Bar) ---
        spectrum_fig = go.Figure()
        harm_labels = ['H1 (60Hz)'] + [f'H{h} ({h * 60}Hz)' for h in HARMONICOS_IMPARES]

        for item in spectrum_data:
            spectrum_fig.add_trace(go.Bar(
                x=harm_labels,
                y=item['Amps'],
                name=item['Caso'],
                marker_color=item['Cor']
            ))
        spectrum_fig.update_layout(title='Espectro Harmônico Completo (Corrente I_800)',
                                   yaxis_title='Amplitude (A)', barmode='group')
        bar_figs['Spectrum_Full'] = spectrum_fig

    return line_figs, bar_figs


# ===================== Interface =====================
uploaded_files = st.file_uploader(
    "Selecione arquivos .mat (da pasta 'resultados_faltas_com_fft')",
    accept_multiple_files=True, type=['.mat']
)

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs = processar_arquivos(tuple(uploaded_files))

    st.success("Processamento concluído!")

    # Abas organizadas
    tabs = st.tabs([
        "📊 Métricas Principais",
        "🌊 Espectro Harmônico",
        "📈 Ponto 800 (Onda/FFT)",
        "📉 Ponto T2F (Onda/FFT)"
    ])

    with tabs[0]:  # Métricas
        if bar_figs:
            st.subheader("Análise de Corrente (Ponto 800)")
            c1, c2 = st.columns(2)
            c1.plotly_chart(bar_figs['Bar_I_Pico'], use_container_width=True)
            c2.plotly_chart(bar_figs['Bar_I_THD'], use_container_width=True)

            st.subheader("Detalhamento de Harmônicas (Corrente)")
            c3, c4, c5 = st.columns(3)
            c3.plotly_chart(bar_figs['Bar_I_h3'], use_container_width=True)
            c4.plotly_chart(bar_figs['Bar_I_h5'], use_container_width=True)
            c5.plotly_chart(bar_figs['Bar_I_h7'], use_container_width=True)

            st.subheader("Análise de Tensão (Ponto 800)")
            c6, c7 = st.columns(2)
            c6.plotly_chart(bar_figs['Bar_V_Pico'], use_container_width=True)
            c7.plotly_chart(bar_figs['Bar_V_h3'], use_container_width=True)

    with tabs[1]:  # Espectro Completo
        if bar_figs:
            st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)
            st.info("Este gráfico mostra a magnitude da Fundamental (H1) e de todas as ímpares até H15 lado a lado.")

    with tabs[2]:  # Ponto 800
        c1, c2 = st.columns(2)
        c1.plotly_chart(line_figs['I800_T'], use_container_width=True)
        c2.plotly_chart(line_figs['I800_F'], use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(line_figs['V800_T'], use_container_width=True)
        c4.plotly_chart(line_figs['V800_F'], use_container_width=True)

    with tabs[3]:  # Ponto T2F
        c1, c2 = st.columns(2)
        c1.plotly_chart(line_figs['IT2F_T'], use_container_width=True)
        c2.plotly_chart(line_figs['IT2F_F'], use_container_width=True)
        c3, c4 = st.columns(2)
        c3.plotly_chart(line_figs['VT2F_T'], use_container_width=True)
        c4.plotly_chart(line_figs['VT2F_F'], use_container_width=True)

    # Download
    st.sidebar.markdown("### 📥 Exportar Dados")
    if st.sidebar.button("Gerar ZIP com Imagens"):
        with st.spinner("Gerando imagens..."):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                all_figs = {**line_figs, **bar_figs}
                for name, fig in all_figs.items():
                    try:
                        fname = FNAMES_MAP.get(name, name)
                        zf.writestr(f"{fname}.png", fig.to_image(format="png", width=1200, height=700))
                        zf.writestr(f"{fname}.html", fig.to_html())
                    except Exception as e:
                        st.error(f"Erro ao salvar {name}: {e}")

            st.sidebar.download_button("Baixar ZIP", zip_buffer.getvalue(), "analise_harmonicas.zip", "application/zip")

else:
    st.info("Por favor, faça upload dos arquivos .mat para iniciar.")