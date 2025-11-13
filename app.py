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
PLOT_XLIM_FFT = [0, 1000]
HARMONICOS_IMPARES = [3, 5, 7, 9, 11, 13, 15]  # Para marcadores no espectro
# Lista dos pontos de medição conforme salvos no MATLAB
PONTOS = ['800', 'T2F', '818', '820', '822']


# ===================== Funções Auxiliares =====================
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    """Encontra a amplitude de uma frequência específica."""
    target_freq = order * fund_freq
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]


def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    """Calcula THD (Total Harmonic Distortion) usando a definição padrão."""
    amp_h1 = get_harmonic_amplitude(freq_array, amp_array, 1, fund_freq)
    if amp_h1 == 0: return 0

    sum_squares = 0
    # Soma os quadrados das harmônicas (inteiras) até max_freq
    for order in range(2, int(max_freq / fund_freq)):
        amp_hn = get_harmonic_amplitude(freq_array, amp_array, order, fund_freq)
        sum_squares += amp_hn ** 2

    thd = (np.sqrt(sum_squares) / amp_h1) * 100  # Em porcentagem
    return thd


# ===================== Função de Processamento =====================
@st.cache_data
def processar_arquivos(uploaded_files):
    numFiles = len(uploaded_files)
    colors = qualitative.Plotly

    # 1. Inicialização Dinâmica das Estruturas de Dados
    legend_entries = []

    # Dicionário para armazenar métricas (colunas do DataFrame futuro)
    # Estrutura: metrics_data['I800_Pico'] = [valor1, valor2...]
    metrics_data = {'Caso': [], 'Cor': []}

    # Cria chaves para todas as métricas de todos os pontos
    metricas_types = ['Pico', 'THD', 'H3', 'H5', 'H7']
    for p in PONTOS:
        for var in ['I', 'V']:  # Corrente e Tensão
            for m in metricas_types:
                metrics_data[f'{var}{p}_{m}'] = []

    spectrum_data = []  # Para o gráfico de espectro completo

    # 2. Inicializa Figuras de Linha (Tempo e FFT) para todos os pontos
    line_figs = {}

    for p in PONTOS:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            key_root = f'{var}{p}'  # Ex: I800, VT2F

            # Figura no Tempo
            line_figs[f'{key_root}_T'] = go.Figure(layout=go.Layout(
                title=f'Sinal no Tempo: {label} ({var}_{p}) - Fase A',
                xaxis_title='Tempo (s)', yaxis_title='Amplitude'))

            # Figura FFT
            line_figs[f'{key_root}_F'] = go.Figure(layout=go.Layout(
                title=f'Espectro FFT: {label} ({var}_{p}) - Fase A',
                xaxis_title='Frequência (Hz)', yaxis_title='Amplitude (dB)',
                xaxis_range=PLOT_XLIM_FFT))

            # Adiciona linhas verticais para harmônicas
            for h in HARMONICOS_IMPARES:
                line_figs[f'{key_root}_F'].add_vline(x=h * F_FUNDAMENTAL, line_width=1, line_dash="dot",
                                                     line_color="gray", opacity=0.5)

    # --- Loop Principal de Arquivos ---
    for i, file in enumerate(uploaded_files):
        matFile = file.name
        currentColor = colors[i % len(colors)]

        try:
            data = sio.loadmat(io.BytesIO(file.read()))
            # Limpeza do nome para legenda
            plotTitle = matFile.replace('.mat', '').replace('_analise_completa', '').replace('__', ' - ').replace('_',
                                                                                                                  ' ')

            legend_entries.append(plotTitle)
            metrics_data['Caso'].append(plotTitle)
            metrics_data['Cor'].append(currentColor)

            # --- Processamento por Ponto ---
            for p in PONTOS:
                # Chaves no .mat (ex: ts_I_800, f_I800)
                # Nota: O script MATLAB salva como ts_I_800, ts_I_T2F, etc.
                suffix = f'_{p}'

                for var in ['I', 'V']:
                    # Nomes das chaves no struct do MATLAB
                    ts_key = f'ts_{var}{suffix}'  # ex: ts_I_800
                    f_key = f'f_{var}{suffix}'  # ex: f_I_800 (cuidado com underscore extra se houver)
                    # No seu script MATLAB, fft_data salva como f_I_800 ou f_I800?
                    # Verificando seu script anterior: tosave.fft_data.(['f_' vn]) onde vn é 'I_800'.
                    # Então a chave é 'f_I_800'.

                    # Ajuste fino para as chaves de FFT baseadas no seu script MATLAB
                    # vn era 'I_800', entao a chave é 'f_I_800'
                    fft_prefix = f'{var}_{p}'  # I_800
                    fft_f_key = f'f_{fft_prefix}'
                    fft_p1_key = f'P1_{fft_prefix}'

                    # Chaves para as Figuras (sem underscore extra)
                    fig_key_root = f'{var}{p}'  # I800

                    try:
                        # Extração dos dados
                        t = data[ts_key][0, 0]['Time'].flatten()
                        y = data[ts_key][0, 0]['Data'][:, 0]  # Fase A
                        f = data['fft_data'][0, 0][fft_f_key].flatten()
                        P1 = data['fft_data'][0, 0][fft_p1_key][:, 0]  # Fase A

                        # Plotagem (Linhas)
                        line_figs[f'{fig_key_root}_T'].add_trace(
                            go.Scatter(x=t, y=y, name=plotTitle, line=dict(color=currentColor)))
                        line_figs[f'{fig_key_root}_F'].add_trace(
                            go.Scatter(x=f, y=20 * np.log10(P1 + 1e-9), name=plotTitle, line=dict(color=currentColor)))

                        # Cálculo de Métricas
                        pico = np.max(np.abs(y))
                        thd = calculate_thd(f, P1)
                        h3 = get_harmonic_amplitude(f, P1, 3)
                        h5 = get_harmonic_amplitude(f, P1, 5)
                        h7 = get_harmonic_amplitude(f, P1, 7)

                        # Armazenamento das métricas
                        metrics_data[f'{fig_key_root}_Pico'].append(pico)
                        metrics_data[f'{fig_key_root}_THD'].append(thd)
                        metrics_data[f'{fig_key_root}_H3'].append(h3)
                        metrics_data[f'{fig_key_root}_H5'].append(h5)
                        metrics_data[f'{fig_key_root}_H7'].append(h7)

                        # Dados para espectro completo (Apenas Corrente do Ponto 800 como exemplo principal, ou todos?)
                        # Vamos pegar I_800 como referência principal de injeção
                        if p == '800' and var == 'I':
                            amps = [get_harmonic_amplitude(f, P1, h) for h in [1] + HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': plotTitle, 'Amps': amps, 'Cor': currentColor})

                    except KeyError as ke:
                        # Se faltar dado de um ponto específico, preenche com 0/NaN para não quebrar o DF
                        # st.warning(f"Dado ausente em {matFile}: {ke}")
                        for m in metricas_types:
                            metrics_data[f'{fig_key_root}_{m}'].append(0)

        except Exception as e:
            st.error(f"Erro fatal no arquivo '{matFile}': {e}")
            continue

    # --- Criação dos Gráficos de Barra (Bar Figs) ---
    bar_figs = {}
    df = pd.DataFrame(metrics_data)

    if not df.empty:
        # Função auxiliar para criar gráfico de barras padronizado
        def create_bar(y_col, title_text, y_label):
            fig = go.Figure(go.Bar(x=df['Caso'], y=df[y_col], marker_color=df['Cor'], name=title_text))
            fig.update_layout(title=title_text, yaxis_title=y_label, xaxis_tickangle=-45)
            return fig

        # Gera barras para TODOS os pontos e métricas
        for p in PONTOS:
            # Corrente
            base_I = f'I{p}'
            bar_figs[f'Bar_{base_I}_Pico'] = create_bar(f'{base_I}_Pico', f'Pico Corrente ({base_I})', 'Amperes')
            bar_figs[f'Bar_{base_I}_THD'] = create_bar(f'{base_I}_THD', f'THD Corrente ({base_I})', '%')
            bar_figs[f'Bar_{base_I}_H3'] = create_bar(f'{base_I}_H3', f'H3 Corrente ({base_I})', 'Amperes')
            bar_figs[f'Bar_{base_I}_H5'] = create_bar(f'{base_I}_H5', f'H5 Corrente ({base_I})', 'Amperes')
            bar_figs[f'Bar_{base_I}_H7'] = create_bar(f'{base_I}_H7', f'H7 Corrente ({base_I})', 'Amperes')

            # Tensão
            base_V = f'V{p}'
            bar_figs[f'Bar_{base_V}_Pico'] = create_bar(f'{base_V}_Pico', f'Pico Tensão ({base_V})', 'Volts')
            bar_figs[f'Bar_{base_V}_THD'] = create_bar(f'{base_V}_THD', f'THD Tensão ({base_V})', '%')
            bar_figs[f'Bar_{base_V}_H3'] = create_bar(f'{base_V}_H3', f'H3 Tensão ({base_V})', 'Volts')
            bar_figs[f'Bar_{base_V}_H5'] = create_bar(f'{base_V}_H5', f'H5 Tensão ({base_V})', 'Volts')
            bar_figs[f'Bar_{base_V}_H7'] = create_bar(f'{base_V}_H7', f'H7 Tensão ({base_V})', 'Volts')

        # Gráfico de Espectro Completo (I_800)
        if spectrum_data:
            spectrum_fig = go.Figure()
            harm_labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]

            for item in spectrum_data:
                spectrum_fig.add_trace(go.Bar(
                    x=harm_labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']
                ))
            spectrum_fig.update_layout(title='Espectro Harmônico Completo (I_800) - Comparação H1..H15',
                                       yaxis_title='Amplitude (A)', barmode='group')
            # Escala logarítmica no eixo Y pode ser útil para ver H altas junto com H1
            # spectrum_fig.update_yaxes(type="log")
            bar_figs['Spectrum_Full'] = spectrum_fig

    return line_figs, bar_figs


# ===================== Interface =====================
uploaded_files = st.file_uploader(
    "Selecione arquivos .mat (da pasta 'resultados_faltas_com_fft_expandido')",
    accept_multiple_files=True, type=['.mat']
)

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs = processar_arquivos(tuple(uploaded_files))

    st.success("Processamento concluído!")

    # --- Criação de Abas Dinâmicas ---
    # 1 Aba de Métricas Gerais + 1 Aba de Espectro + 1 Aba por Ponto
    tab_names = ["📊 Métricas (Barras)", "🌊 Espectro Detalhado"] + [f"📍 Ponto {p}" for p in PONTOS]
    tabs = st.tabs(tab_names)

    # 1. Aba de Métricas (Organizada por Expander)
    with tabs[0]:
        st.markdown("### Comparação Quantitativa")

        # Espectro Completo no topo
        if 'Spectrum_Full' in bar_figs:
            st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)
            st.markdown("---")

        # Expander para cada ponto
        for p in PONTOS:
            with st.expander(f"Métricas do Ponto {p}", expanded=(p == '800')):
                st.markdown(f"#### Corrente ({p})")
                c1, c2 = st.columns(2)
                c1.plotly_chart(bar_figs[f'Bar_I{p}_Pico'], use_container_width=True)
                c2.plotly_chart(bar_figs[f'Bar_I{p}_THD'], use_container_width=True)

                c3, c4, c5 = st.columns(3)
                c3.plotly_chart(bar_figs[f'Bar_I{p}_H3'], use_container_width=True)
                c4.plotly_chart(bar_figs[f'Bar_I{p}_H5'], use_container_width=True)
                c5.plotly_chart(bar_figs[f'Bar_I{p}_H7'], use_container_width=True)

                st.markdown(f"#### Tensão ({p})")
                c6, c7 = st.columns(2)
                c6.plotly_chart(bar_figs[f'Bar_V{p}_Pico'], use_container_width=True)
                c7.plotly_chart(bar_figs[f'Bar_V{p}_THD'], use_container_width=True)

    # 2. Aba Extra (Placeholder ou Info)
    with tabs[1]:
        st.info(
            "A aba 'Métricas' acima já contém a análise de H3, H5 e H7. O gráfico de espectro completo (H1-H15) também está lá.")

    # 3. Abas dos Pontos (Formas de Onda)
    # As abas de pontos começam no índice 2 da lista 'tabs'
    for i, p in enumerate(PONTOS):
        with tabs[i + 2]:
            st.markdown(f"### Formas de Onda e FFT - Ponto {p}")
            c1, c2 = st.columns(2)
            c1.plotly_chart(line_figs[f'I{p}_T'], use_container_width=True)
            c2.plotly_chart(line_figs[f'I{p}_F'], use_container_width=True)

            c3, c4 = st.columns(2)
            c3.plotly_chart(line_figs[f'V{p}_T'], use_container_width=True)
            c4.plotly_chart(line_figs[f'V{p}_F'], use_container_width=True)

    # --- Download ---
    st.sidebar.markdown("### 📥 Exportar Tudo")
    if st.sidebar.button("Gerar ZIP (Todas as Imagens)"):
        with st.spinner("Gerando dezenas de imagens... (Isso pode demorar)"):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Junta tudo
                all_figs = {**line_figs, **bar_figs}

                for name, fig in all_figs.items():
                    try:
                        # Nome do arquivo limpo
                        fname = f"{name}.png"
                        zf.writestr(fname, fig.to_image(format="png", width=1200, height=700))
                    except Exception as e:
                        st.sidebar.error(f"Erro ao salvar {name}: {e}")

            st.sidebar.download_button("Baixar ZIP Completo", zip_buffer.getvalue(), "analise_completa_5pontos.zip",
                                       "application/zip")

else:
    st.info("Faça upload dos arquivos .mat gerados pelo script MATLAB expandido.")