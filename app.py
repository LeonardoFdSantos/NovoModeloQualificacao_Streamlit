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
st.title("Analisador Avançado de Qualidade de Energia (Espectro até 10kHz)")

# ===================== Constantes =====================
F_FUNDAMENTAL = 60  # Hz
F_MAX_ANALISE = 10000  # 10 kHz

# Gera lista de harmônicas ímpares automaticamente até 10 kHz
# Ex: [3, 5, 7, ..., 165, 167]
HARMONICOS_IMPARES = [h for h in range(3, int(F_MAX_ANALISE / F_FUNDAMENTAL) + 1, 2)]

# Lista dos pontos de medição
PONTOS = ['800', 'T2F', '818', '820', '822']


# ===================== Funções Auxiliares =====================
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    """Encontra a amplitude de uma frequência específica."""
    target_freq = order * fund_freq
    # Encontra o índice mais próximo da frequência alvo
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]


def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    """Calcula THD considerando harmônicas até max_freq."""
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

    # Inicializa colunas do DataFrame
    metricas_types = ['Pico', 'THD', 'H3', 'H5', 'H7']
    for p in PONTOS:
        for var in ['I', 'V']:
            for m in metricas_types:
                metrics_data[f'{var}{p}_{m}'] = []

    spectrum_data = []

    # --- 1. Inicializa Figuras de Linha ---
    line_figs = {}

    for p in PONTOS:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            key_root = f'{var}{p}'

            # Figura Tempo (Autoscale padrão do Plotly)
            line_figs[f'{key_root}_T'] = go.Figure(layout=go.Layout(
                title=f'Sinal no Tempo: {label} ({var}_{p}) - Fase A',
                xaxis_title='Tempo (s)', yaxis_title='Amplitude'))

            # Figura FFT (Autoscale + Faixa inicial até 10k)
            line_figs[f'{key_root}_F'] = go.Figure(layout=go.Layout(
                title=f'Espectro FFT: {label} ({var}_{p}) - Fase A',
                xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)',
                # Define o range inicial, mas permite zoom out (autoscale) pelo usuário
                xaxis=dict(range=[0, F_MAX_ANALISE], autorange=False)
            ))

            # Adiciona marcadores para TODAS as harmônicas ímpares
            for h in HARMONICOS_IMPARES:
                # Linha vertical mais sutil para não poluir
                line_figs[f'{key_root}_F'].add_vline(
                    x=h * F_FUNDAMENTAL,
                    line_width=0.5,
                    line_dash="dot",
                    line_color="rgba(128, 128, 128, 0.3)"  # Cinza transparente
                )

                # Adiciona texto APENAS para as primeiras harmônicas para legibilidade
                if h <= 15:
                    line_figs[f'{key_root}_F'].add_annotation(
                        x=h * F_FUNDAMENTAL, y=0, text=f"H{h}",
                        showarrow=False, yshift=5, font=dict(size=8, color="gray")
                    )

    # --- Loop Principal de Arquivos ---
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
                    clean_vn = vn.replace('_', '')  # Ex: I800

                    field_ts = f'ts_{vn}'
                    field_f = f'f_{vn}'
                    field_p1 = f'P1_{vn}'

                    fig_key_root = clean_vn

                    try:
                        # Extração
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y = ts_data_struct['Data'][:, 0]  # Fase A
                        f = struct_fft[field_f].flatten()
                        P1 = struct_fft[field_p1][:, 0]  # Fase A

                        # Plotagem
                        line_figs[f'{fig_key_root}_T'].add_trace(
                            go.Scatter(x=t, y=y, name=plotTitle, line=dict(color=currentColor)))

                        # FFT em dB
                        line_figs[f'{fig_key_root}_F'].add_trace(
                            go.Scatter(x=f, y=20 * np.log10(P1 + 1e-9), name=plotTitle, line=dict(color=currentColor)))

                        # Métricas
                        pico = np.max(np.abs(y))
                        # THD calculado até a frequência máxima de análise
                        thd = calculate_thd(f, P1, fund_freq=F_FUNDAMENTAL, max_freq=F_MAX_ANALISE)
                        h3 = get_harmonic_amplitude(f, P1, 3)
                        h5 = get_harmonic_amplitude(f, P1, 5)
                        h7 = get_harmonic_amplitude(f, P1, 7)

                        metrics_data[f'{fig_key_root}_Pico'].append(pico)
                        metrics_data[f'{fig_key_root}_THD'].append(thd)
                        metrics_data[f'{fig_key_root}_H3'].append(h3)
                        metrics_data[f'{fig_key_root}_H5'].append(h5)
                        metrics_data[f'{fig_key_root}_H7'].append(h7)

                        # Espectro Completo (Usando I_T2F)
                        if p == 'T2F' and var == 'I':
                            # Pega amplitudes de H1 até o máximo da lista
                            amps = [get_harmonic_amplitude(f, P1, h) for h in [1] + HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': plotTitle, 'Amps': amps, 'Cor': currentColor})

                    except ValueError:
                        for m in metricas_types:
                            metrics_data[f'{fig_key_root}_{m}'].append(0)

        except Exception as e:
            st.error(f"Erro fatal no arquivo '{matFile}': {e}")
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

        # Gráfico de Espectro Completo (I_T2F até 10k)
        if spectrum_data:
            spectrum_fig = go.Figure()
            # Rótulos para todas as harmônicas (vai ficar denso, mas é o pedido)
            harm_labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]

            for item in spectrum_data:
                spectrum_fig.add_trace(go.Bar(
                    x=harm_labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']
                ))

            spectrum_fig.update_layout(
                title=f'Espectro Harmônico Completo (I_T2F) - H1 até ~H{HARMONICOS_IMPARES[-1]} ({F_MAX_ANALISE} Hz)',
                yaxis_title='Amplitude (A)',
                barmode='group',
                xaxis_tickangle=-90  # Rótulos verticais para caberem todos
            )
            bar_figs['Spectrum_Full'] = spectrum_fig

    return line_figs, bar_figs


# ===================== Interface =====================
uploaded_files = st.file_uploader(
    "Selecione arquivos .mat",
    accept_multiple_files=True, type=['.mat']
)

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs = processar_arquivos(tuple(uploaded_files))

    st.success("Concluído!")

    tab_names = ["📊 Métricas", "🌊 Espectro (I_T2F)", "📈 Ponto 800", "📉 Ponto T2F"] + [f"📍 {p}" for p in
                                                                                      ['818', '820', '822']]
    tabs = st.tabs(tab_names)

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

    with tabs[1]:
        if 'Spectrum_Full' in bar_figs:
            st.markdown("#### Varredura Espectral Completa (Ímpares até 10 kHz)")
            st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

    for i, p in enumerate(PONTOS):
        with tabs[i + 2]:
            st.markdown(f"### Análise Detalhada: Ponto {p}")
            c1, c2 = st.columns(2)
            c1.plotly_chart(line_figs[f'I{p}_T'], use_container_width=True)
            c2.plotly_chart(line_figs[f'I{p}_F'], use_container_width=True)

            c3, c4 = st.columns(2)
            c3.plotly_chart(line_figs[f'V{p}_T'], use_container_width=True)
            c4.plotly_chart(line_figs[f'V{p}_F'], use_container_width=True)

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