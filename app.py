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
HARMONICOS_IMPARES = [3, 5, 7, 9, 11, 13, 15]
# Lista dos pontos exatos conforme nomes no MATLAB (800, T2F, 818, etc)
PONTOS = ['800', 'T2F', '818', '820', '822']


# ===================== Funções Auxiliares =====================
def get_harmonic_amplitude(freq_array, amp_array, order, fund_freq=60):
    """Encontra a amplitude de uma frequência específica."""
    target_freq = order * fund_freq
    # Encontra o índice mais próximo da frequência alvo
    idx = (np.abs(freq_array - target_freq)).argmin()
    return amp_array[idx]


def calculate_thd(freq_array, amp_array, fund_freq=60, max_freq=2000):
    """Calcula THD (Total Harmonic Distortion) usando os dados da FFT."""
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

    # Estruturas para armazenar dados para gráficos de barra
    legend_entries = []
    metrics_data = {'Caso': [], 'Cor': []}

    # Inicializa colunas do DataFrame de métricas
    metricas_types = ['Pico', 'THD', 'H3', 'H5', 'H7']
    for p in PONTOS:
        for var in ['I', 'V']:
            for m in metricas_types:
                metrics_data[f'{var}{p}_{m}'] = []

    spectrum_data = []

    # --- Inicializa Figuras de Linha ---
    line_figs = {}
    for p in PONTOS:
        for var, label in [('I', 'Corrente'), ('V', 'Tensão')]:
            # Chave limpa para o dicionário de figuras (ex: I800, VT2F)
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

            # Marcadores de harmônicas
            for h in HARMONICOS_IMPARES:
                line_figs[f'{fig_key}_F'].add_vline(x=h * F_FUNDAMENTAL, line_width=1, line_dash="dot",
                                                    line_color="gray", opacity=0.5)

    # --- Loop Principal de Arquivos ---
    for i, file in enumerate(uploaded_files):
        matFile = file.name
        currentColor = colors[i % len(colors)]

        try:
            # Carrega o arquivo .mat
            data = sio.loadmat(io.BytesIO(file.read()))

            # Nome limpo para legenda
            plotTitle = matFile.replace('.mat', '').replace('__', ' - ').replace('_', ' ')
            legend_entries.append(plotTitle)
            metrics_data['Caso'].append(plotTitle)
            metrics_data['Cor'].append(currentColor)

            # --- Extração das Estruturas Principais ---
            # O scipy carrega structs do MATLAB como arrays numpy aninhados [0,0]
            # Precisamos acessar a raiz de cada struct salvo no MATLAB
            try:
                struct_ts = data['ts'][0, 0]
                struct_fft = data['fft_data'][0, 0]
                # analise -> Picos
                struct_analise_picos = data['analise'][0, 0]['Picos'][0, 0]
            except KeyError:
                st.error(
                    f"Estrutura inválida no arquivo {matFile}. Verifique se foi gerado pelo script MATLAB correto.")
                continue

            # --- Processamento por Ponto (800, T2F, 818...) ---
            for p in PONTOS:
                for var in ['I', 'V']:
                    # Constrói os nomes das chaves conforme salvo no MATLAB
                    # Ex: var='I', p='800' -> vn='I_800'
                    vn = f"{var}_{p}"
                    clean_vn = vn.replace('_', '')  # Ex: I800

                    # Nomes dos campos dentro dos structs
                    field_ts = f'ts_{vn}'  # ex: ts_I_800
                    field_f = f'f_{vn}'  # ex: f_I_800
                    field_p1 = f'P1_{vn}'  # ex: P1_I_800

                    fig_key = clean_vn  # Chave para nossos dicionários locais (I800)

                    try:
                        # 1. Extrair Dados de Tempo
                        # Acesso: struct_ts[field_ts][0,0] -> pegando o conteúdo
                        ts_data_struct = struct_ts[field_ts][0, 0]
                        t = ts_data_struct['Time'].flatten()
                        y = ts_data_struct['Data'][:, 0]  # Fase A

                        # 2. Extrair Dados de FFT
                        f = struct_fft[field_f].flatten()
                        P1 = struct_fft[field_p1][:, 0]  # Fase A

                        # 3. Plotar
                        line_figs[f'{fig_key}_T'].add_trace(
                            go.Scatter(x=t, y=y, name=plotTitle, line=dict(color=currentColor)))
                        line_figs[f'{fig_key}_F'].add_trace(
                            go.Scatter(x=f, y=20 * np.log10(P1 + 1e-9), name=plotTitle, line=dict(color=currentColor)))

                        # 4. Calcular Métricas (No Python)
                        # Nota: Podemos usar os valores pré-calculados do MATLAB em 'struct_analise_picos'
                        # ou recalcular aqui. Recalcular no Python com os dados brutos da FFT
                        # garante que THD, H5 e H7 estejam consistentes.

                        pico = np.max(np.abs(y))
                        thd = calculate_thd(f, P1)
                        h3 = get_harmonic_amplitude(f, P1, 3)
                        h5 = get_harmonic_amplitude(f, P1, 5)
                        h7 = get_harmonic_amplitude(f, P1, 7)

                        # Salva nas listas
                        metrics_data[f'{fig_key}_Pico'].append(pico)
                        metrics_data[f'{fig_key}_THD'].append(thd)
                        metrics_data[f'{fig_key}_H3'].append(h3)
                        metrics_data[f'{fig_key}_H5'].append(h5)
                        metrics_data[f'{fig_key}_H7'].append(h7)

                        # Dados especiais para espectro completo (Apenas Corrente I_800)
                        if p == '800' and var == 'I':
                            amps = [get_harmonic_amplitude(f, P1, h) for h in [1] + HARMONICOS_IMPARES]
                            spectrum_data.append({'Caso': plotTitle, 'Amps': amps, 'Cor': currentColor})

                    except ValueError as ve:
                        # Caso o campo não exista (ex: nome errado), preenche com 0
                        # st.warning(f"Aviso em {matFile}: Campo {field_ts} não encontrado.")
                        for m in metricas_types:
                            metrics_data[f'{fig_key}_{m}'].append(0)

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
            # I
            base = f'I{p}'
            bar_figs[f'{base}_Pico'] = create_bar(f'{base}_Pico', f'Pico Corrente ({base})', 'A')
            bar_figs[f'{base}_THD'] = create_bar(f'{base}_THD', f'THD Corrente ({base})', '%')
            bar_figs[f'{base}_H3'] = create_bar(f'{base}_H3', f'H3 Corrente ({base})', 'A')
            bar_figs[f'{base}_H5'] = create_bar(f'{base}_H5', f'H5 Corrente ({base})', 'A')
            bar_figs[f'{base}_H7'] = create_bar(f'{base}_H7', f'H7 Corrente ({base})', 'A')
            # V
            base = f'V{p}'
            bar_figs[f'{base}_Pico'] = create_bar(f'{base}_Pico', f'Pico Tensão ({base})', 'V')
            bar_figs[f'{base}_THD'] = create_bar(f'{base}_THD', f'THD Tensão ({base})', '%')
            bar_figs[f'{base}_H3'] = create_bar(f'{base}_H3', f'H3 Tensão ({base})', 'V')
            bar_figs[f'{base}_H5'] = create_bar(f'{base}_H5', f'H5 Tensão ({base})', 'V')
            bar_figs[f'{base}_H7'] = create_bar(f'{base}_H7', f'H7 Tensão ({base})', 'V')

        # Espectro Completo
        if spectrum_data:
            sp_fig = go.Figure()
            labels = ['H1'] + [f'H{h}' for h in HARMONICOS_IMPARES]
            for item in spectrum_data:
                sp_fig.add_trace(go.Bar(x=labels, y=item['Amps'], name=item['Caso'], marker_color=item['Cor']))
            sp_fig.update_layout(title='Espectro Harmônico (I_800) - H1 a H15', yaxis_title='Amplitude (A)',
                                 barmode='group')
            # sp_fig.update_yaxes(type="log") # Opcional: Log para ver harmônicas pequenas
            bar_figs['Spectrum_Full'] = sp_fig

    return line_figs, bar_figs


# ===================== Interface =====================
uploaded_files = st.file_uploader(
    "Selecione arquivos .mat (da pasta 'resultados_faltas_com_fft_expandido')",
    accept_multiple_files=True, type=['.mat']
)

if uploaded_files:
    with st.spinner('Processando...'):
        line_figs, bar_figs = processar_arquivos(tuple(uploaded_files))

    st.success("Concluído!")

    # Abas
    tab_names = ["📊 Métricas", "🌊 Espectro", "📈 Ponto 800", "📉 Ponto T2F"] + [f"📍 {p}" for p in ['818', '820', '822']]
    tabs = st.tabs(tab_names)

    # 1. Métricas
    with tabs[0]:
        st.markdown("### Comparação de Harmônicas e THD")
        if 'Spectrum_Full' in bar_figs:
            st.plotly_chart(bar_figs['Spectrum_Full'], use_container_width=True)

        for p in PONTOS:
            with st.expander(f"Dados do Ponto {p}", expanded=(p == '800')):
                c1, c2, c3 = st.columns(3)
                c1.plotly_chart(bar_figs[f'I{p}_Pico'], use_container_width=True)
                c2.plotly_chart(bar_figs[f'I{p}_THD'], use_container_width=True)
                c3.plotly_chart(bar_figs[f'I{p}_H3'], use_container_width=True)

                c4, c5, c6 = st.columns(3)
                c4.plotly_chart(bar_figs[f'V{p}_Pico'], use_container_width=True)
                c5.plotly_chart(bar_figs[f'V{p}_THD'], use_container_width=True)
                c6.plotly_chart(bar_figs[f'V{p}_H3'], use_container_width=True)

    # 2. Espectro (Info)
    with tabs[1]:
        st.info("Veja o gráfico 'Espectro Harmônico' na aba Métricas.")

    # 3. Gráficos de Onda por Ponto
    # Mapeia as abas para os pontos
    pontos_abas = ['800', 'T2F', '818', '820', '822']
    for i, p in enumerate(pontos_abas):
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