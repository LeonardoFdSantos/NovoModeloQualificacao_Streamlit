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
PLOT_XLIM_FFT = [0, 1000]  # Visualizar até 1kHz (aprox H16)
HARMONICOS_IMPARES = [3, 5, 7, 9, 11, 13, 15]
# Lista dos pontos exatos conforme nomes no MATLAB
PONTOS = ['800', 'T2F', '818', '820', '822']


# ===================== Funções Auxiliares =====================
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    target_freq = order * fund_freq
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]


def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    amp_h1 = get_harmonic_amplitude(freq_array, amp_array, 1, fund_freq)
    if amp_h1 == 0: return 0

    sum_squares = 0
    for order in range(2, int(max_freq / fund_freq)):
        amp_hn = get_harmonic_amplitude(freq_array, amp_array, order, fund_freq)
        sum_squares += amp_hn ** 2

    thd = (np.sqrt(sum_squares) / amp_h1) * 100
    return thd


# ===================== Função de Processamento =====================
@st.cache_data
def processar_arquivos(uploaded_files):
    colors = qualitative.Plotly
    legend_entries = []
    metrics_data = {'Caso': [], 'Cor': []}

    metricas_types = ['Pico', 'THD', 'H3', 'H5', 'H7']
    for p in PONTOS:
        for var in ['I', 'V']:
            for m in metricas_types:
                metrics_data[f'{var}{p}_{m}'] = []

    spectrum_data = []

    # --- 1. Inicializa Figuras de Linha (Com Marcadores de Harmônicas) ---
    line_figs = {}
    for p in PONTOS:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            fig_key = f'{var}{p}'

            # Figura Tempo
            line_figs[f'{fig_key}_T'] = go.Figure(layout=go.Layout(
                title=f'Sinal no Tempo: {label} ({var}_{p}) - Fase A',
                xaxis_title='Tempo (s)', yaxis_title='Amplitude'))

            # Figura FFT
            line_figs[f'{fig_key}_F'] = go.Figure(layout=go.Layout(
                title=f'Espectro FFT: {label} ({var}_{p}) - Fase A',
                xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)',
                xaxis_range=PLOT_XLIM_FFT))

            # --- ADIÇÃO: Marcadores de Harmônicas Ímpares ---
            for h in HARMONICOS_IMPARES:
                # Linha Vertical
                line_figs[f'{fig_key}_F'].add_vline(
                    x=h * F_FUNDAMENTAL,
                    line_width=1,
                    line_dash="dot",
                    line_color="gray",
                    opacity=0.5
                )
                # Rótulo de Texto (Apenas para H3, H5, H7, H9 para não poluir)
                if h <= 9:
                    line_figs[f'{fig_key}_F'].add_annotation(
                        x=h * F_FUNDAMENTAL,
                        y=1,  # Posição relativa (topo do gráfico)
                        yref="paper",
                        text=f"H{h}",
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        yshift=10
                    )

    # --- 2. Loop Principal de Arquivos ---
    for i, file in enumerate(uploaded_files):
        matFile = file.name
        currentColor = colors[i % len(colors)]

        try:
            data = sio.loadmat(io.BytesIO(file.read()))
            plotTitle = matFile.replace('.mat', '').replace('__', ' - ').replace('_', ' ')

            legend_entries.append(plotTitle)
            metrics_data['Caso'].append(plotTitle)
            metrics_data['Cor'].append(currentColor)

            # Verifica estrutura (structs aninhados)
            try:
                struct_ts = data['ts'][0, 0]
                struct_fft = data['fft_data'][0, 0]
            except KeyError:
                st.error(f"Estrutura inválida no arquivo {matFile}.")
                continue

            # --- Processamento por Ponto ---
            for p in PONTOS:
                for var in ['I', 'V']:
                    vn = f"{var}_{p}"
                    clean_vn = vn.replace('_', '')  # I800

                    # Nomes das chaves no MATLAB
                    field_ts = f'ts_{vn}'
                    field_f = f'f_{vn}'
                    field_p1 = f'P1_{vn}'

                    fig_key = clean_vn

                    try:
                        # Extração
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y = ts_data_struct['Data'][:, 0]  # Fase A
                        f = struct_fft[field_f].flatten()
                        P1 = struct_fft[field_p1][:, 0]  # Fase A

                        # Plotagem
                        line_figs[f'{fig_key}_T'].add_trace(
                            go.Scatter(x=t, y=y, name=plotTitle, line=dict(color=currentColor)))
                        line_figs[f'{fig_key}_F'].add_trace(
                            go.Scatter(x=f, y=20 * np.log10(P1 + 1e-9), name=plotTitle, line=dict(color=currentColor)))

                        # Métricas (Recalculadas no Python)
                        pico = np.max(np.abs(y))
                        thd = calculate_thd(f, P1)
                        h3 = get_harmonic_amplitude(f, P1, 3)
                        h5 = get_harmonic_amplitude(f, P1, 5)
                        h7 = get_harmonic_amplitude(f, P1, 7)

                        metrics_data[f'{fig_key}_Pico'].append(pico)
                        metrics_data[f'{fig_key}_THD'].append(thd)
                        metrics_data[f'{fig_key}_H3'].append(h3)
                        metrics_data[f'{fig_key}_H5'].append(h5)
                        metrics_data[f'{fig_key}_H7'].append(h7)

                        # Espectro Completo (Usando I_T2F)
                        if p == 'T2F' and var == 'I':
                            amps = [get_harmonic_amplitude(f, P1, h) for h in [1] + HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': plotTitle, 'Amps': amps, 'Cor': currentColor})

                    except ValueError:
                        for m in metricas_types: metrics_data[f'{fig_key}_{m}'].append(0)

        except Exception as e:
            st.error(f"Erro fatal ao ler {matFile}: {e}")
            continue

    # --- 3. Criação dos Gráficos de Barra ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)

    if not df.empty:
        def create_bar(y_col, title_text, y_label):
            fig = go.Figure(go.Bar(x=df['Caso'], y=df[y_col], marker_color=df['Cor'], name=title_text))
            fig.update_layout(title=title_text, yaxis_title=y_label, xaxis_tickangle=-45)
            return fig

        for p in PONTOS:
            base_I = f'I{p}'
            bar_figs[f'Bar_{base_I}_Pico'] = create_bar(f'{base_I}_Pico', f'Pico Corrente ({base_I})', 'A')
            bar_figs[f'Bar_{base_I}_THD'] = create_bar(f'{base_I}_THD', f'THD Corrente ({base_I})', '%')
            bar_figs[f'Bar_{base_I}_H3'] = create_bar(f'{base_I}_H3', f'H3 Corrente ({base_I})', 'A')
            bar_figs[f'Bar_{base_I}_H5'] = create_bar(f'{base_I}_H5', f'H5 Corrente ({base_I})', 'A')
            bar_figs[f'Bar_{base_I}_H7'] = create_bar(f'{base_I}_H7', f'H7 Corrente ({base_I})', 'A')

            base_V = f'V{p}'
            bar_figs[f'Bar_{base_V}_Pico'] = create_bar(f'{base_V}_Pico', f'Pico Tensão ({base_V})', 'V')
            bar_figs[f'Bar_{base_V}_THD'] = create_bar(f'{base_V}_THD', f'THD Tensão ({base_V})', '%')
            bar_figs[f'Bar_{base_V}_H3'] = create_bar(f'{base_V}_H3', f'H3 Tensão ({base_V})', 'V')
            bar_figs[f'Bar_{base_V}_H5'] = create_bar(f'{base_V}_H5', f'H5 Tensão ({base_V})', 'V')
            bar_figs[f'Bar_{base_V}_H7'] = create_bar(f'{base_V}_H7', f'H7 Tensão ({base_V})', 'V')

        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1 (60Hz)'] + [f'H{h} ({h * 60}Hz)' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title='Espectro Harmônico (I_T2F) - H1 a H15', yaxis_title='Amplitude (A)',
                                 barmode='group')
            bar_figs['Spectrum_Full'] = sp_fig

    return line_figs, bar_figs


# ===================== Interface =====================
uploaded_files = st.file_uploader("Selecione arquivos .mat", accept_multiple_files=True, type=['.mat'])

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs = processar_arquivos(tuple(uploaded_files))
    st.success("Concluído!")

    tab_names = ["📊 Métricas", "🌊 Espectro (I_T2F)"] + [f"📍 {p}" for p in PONTOS]
    tabs = st.tabs(tab_names)

    # Aba 1: Métricas
    with tabs[0]:
        st.markdown("### Comparação de Harmônicas e THD")
        for p in PONTOS:
            with st.expander(f"Dados do Ponto {p}", expanded=(p == 'T2F')):
                c1, c2, c3 = st.columns(3)
                c1.plotly_chart(bar_figs[f'Bar_I{p}_Pico'], use_container_width=True)
                c2.plotly_chart(bar_figs[f'Bar_I{p}_THD'], use_container_width=True)
                c3.plotly_chart(bar_figs[f'Bar_I{p}_H3'], use_container_width=True)

                c4, c5, c6 = st.columns(3)
                c4.plotly_chart(bar_figs[f'Bar_V{p}_Pico'], use_container_width=True)
                c5.plotly_chart(bar_figs[f'Bar_V{p}_THD'], use_container_width=True)
                c6.plotly_chart(bar_figs[f'Bar_V{p}_H3'], use_container_width=True)

    # Aba 2: Espectro
    with tabs[1]:
        if 'Spectrum_Full' in bar_figs: st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

    # Abas de Pontos (2 a 6)
    for i, p in enumerate(PONTOS):
        with tabs[i + 2]:
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
                    except:
                        pass
            st.sidebar.download_button("Baixar ZIP", zip_buffer.getvalue(), "resultados.zip", "application/zip")
else:
    st.info("Faça upload dos arquivos .mat.")