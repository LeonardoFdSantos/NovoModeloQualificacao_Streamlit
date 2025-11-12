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
# Define o layout da página para ser "wide" (tela cheia)
st.set_page_config(layout="wide")
st.title("Analisador Comparativo de Simulações (MATLAB)")

# ===================== Constantes =====================
f_fundamental = 60  # Hz
f_h3 = 3 * f_fundamental
plot_xlim_fft = [0, 1000]

# Mapeia os nomes das figuras para os nomes dos arquivos
FNAMES_MAP = {
    'I800_T': 'Comparacao_I800_Tempo', 'I800_F': 'Comparacao_I800_FFT',
    'V800_T': 'Comparacao_V800_Tempo', 'V800_F': 'Comparacao_V800_FFT',
    'I816_T': 'Comparacao_I816_Tempo', 'I816_F': 'Comparacao_I816_FFT',
    'V816_T': 'Comparacao_V816_Tempo', 'V816_F': 'Comparacao_V816_FFT',
    'Bar_I_Pico': 'Comparacao_Bar_I_Pico_800',
    'Bar_I_h3': 'Comparacao_Bar_I_h3_800',
    'Bar_V_Pico': 'Comparacao_Bar_V_Pico_800',
    'Bar_V_h3': 'Comparacao_Bar_V_h3_800'
}


# ===================== Função de Processamento (com Cache) =====================
# Esta é a função principal que processa os arquivos.
# @st.cache_data "salva" o resultado. Se você carregar os mesmos arquivos
# novamente, o Streamlit usará os resultados salvos em vez de reprocessar.

@st.cache_data
def processar_arquivos(uploaded_files):
    """
    Lê os arquivos .mat carregados, processa os dados e gera
    todas as 12 figuras do Plotly.
    """

    numFiles = len(uploaded_files)
    colors = qualitative.Plotly

    # Dicionários para armazenar os dados dos gráficos
    legend_entries = []
    bar_labels = []
    bar_data_I_Pico = []
    bar_data_I_h3 = []
    bar_data_V_Pico = []
    bar_data_V_h3 = []

    # --- 1. Cria os 8 objetos de figura de linha (vazios) ---
    line_figs = {}
    line_figs['I800_T'] = go.Figure(
        layout=go.Layout(title='Sinal no Tempo: Corrente (I_800) - Fase A', xaxis_title='Tempo (s)',
                         yaxis_title='Corrente (A)'))
    line_figs['I800_F'] = go.Figure(
        layout=go.Layout(title='FFT: Corrente (I_800) - Fase A', xaxis_title='Frequência (Hz)',
                         yaxis_title='Amplitude (dB)', xaxis_range=plot_xlim_fft))
    line_figs['I800_F'].add_vline(x=f_h3, line_width=2, line_dash="dash", line_color="red", annotation_text="H3")

    line_figs['V800_T'] = go.Figure(
        layout=go.Layout(title='Sinal no Tempo: Tensão (V_800) - Fase A', xaxis_title='Tempo (s)',
                         yaxis_title='Tensão (V)'))
    line_figs['V800_F'] = go.Figure(
        layout=go.Layout(title='FFT: Tensão (V_800) - Fase A', xaxis_title='Frequência (Hz)',
                         yaxis_title='Amplitude (dB)', xaxis_range=plot_xlim_fft))
    line_figs['V800_F'].add_vline(x=f_h3, line_width=2, line_dash="dash", line_color="red", annotation_text="H3")

    line_figs['I816_T'] = go.Figure(
        layout=go.Layout(title='Sinal no Tempo: Corrente (I_816) - Fase A', xaxis_title='Tempo (s)',
                         yaxis_title='Corrente (A)'))
    line_figs['I816_F'] = go.Figure(
        layout=go.Layout(title='FFT: Corrente (I_816) - Fase A', xaxis_title='Frequência (Hz)',
                         yaxis_title='Amplitude (dB)', xaxis_range=plot_xlim_fft))
    line_figs['I816_F'].add_vline(x=f_h3, line_width=2, line_dash="dash", line_color="red", annotation_text="H3")

    line_figs['V816_T'] = go.Figure(
        layout=go.Layout(title='Sinal no Tempo: Tensão (V_816) - Fase A', xaxis_title='Tempo (s)',
                         yaxis_title='Tensão (V)'))
    line_figs['V816_F'] = go.Figure(
        layout=go.Layout(title='FFT: Tensão (V_816) - Fase A', xaxis_title='Frequência (Hz)',
                         yaxis_title='Amplitude (dB)', xaxis_range=plot_xlim_fft))

    for fig in line_figs.values():
        fig.update_layout(hovermode="x unified", yaxis_gridcolor='#eee', xaxis_gridcolor='#eee',
                          legend_title_text='Casos Comparados')

    # --- 2. Loop principal: Carrega cada arquivo e adiciona traços ---
    for i, file in enumerate(uploaded_files):
        matFile = file.name
        currentColor = colors[i % len(colors)]

        try:
            # Lê o arquivo .mat da memória
            data = sio.loadmat(io.BytesIO(file.read()))

            # Prepara o nome da legenda
            plotTitle = matFile.replace('.mat', '').replace('_analise_completa', '').replace('__', ' - ').replace('_',
                                                                                                                  ' ')

            # Extrai os dados (idêntico ao seu script)
            ts_I_800_time = data['ts_I_800'][0, 0]['Time'].flatten()
            ts_I_800_data = data['ts_I_800'][0, 0]['Data'][:, 0]
            f_I800 = data['fft_data'][0, 0]['f_I800'].flatten()
            P1_I800 = data['fft_data'][0, 0]['P1_I800'][:, 0]

            ts_V_800_time = data['ts_V_800'][0, 0]['Time'].flatten()
            ts_V_800_data = data['ts_V_800'][0, 0]['Data'][:, 0]
            f_V800 = data['fft_data'][0, 0]['f_V800'].flatten()
            P1_V800 = data['fft_data'][0, 0]['P1_V800'][:, 0]

            ts_I_816_time = data['ts_I_816'][0, 0]['Time'].flatten()
            ts_I_816_data = data['ts_I_816'][0, 0]['Data'][:, 0]
            f_I816 = data['fft_data'][0, 0]['f_I816'].flatten()
            P1_I816 = data['fft_data'][0, 0]['P1_I816'][:, 0]

            ts_V_816_time = data['ts_V_816'][0, 0]['Time'].flatten()
            ts_V_816_data = data['ts_V_816'][0, 0]['Data'][:, 0]
            f_V816 = data['fft_data'][0, 0]['f_V816'].flatten()
            P1_V816 = data['fft_data'][0, 0]['P1_V816'][:, 0]

            # Adiciona os traços (traces) às figuras
            line_figs['I800_T'].add_trace(
                go.Scatter(x=ts_I_800_time, y=ts_I_800_data, name=plotTitle, line=dict(color=currentColor)))
            line_figs['I800_F'].add_trace(
                go.Scatter(x=f_I800, y=20 * np.log10(P1_I800), name=plotTitle, line=dict(color=currentColor)))
            line_figs['V800_T'].add_trace(
                go.Scatter(x=ts_V_800_time, y=ts_V_800_data, name=plotTitle, line=dict(color=currentColor)))
            line_figs['V800_F'].add_trace(
                go.Scatter(x=f_V800, y=20 * np.log10(P1_V800), name=plotTitle, line=dict(color=currentColor)))
            line_figs['I816_T'].add_trace(
                go.Scatter(x=ts_I_816_time, y=ts_I_816_data, name=plotTitle, line=dict(color=currentColor)))
            line_figs['I816_F'].add_trace(
                go.Scatter(x=f_I816, y=20 * np.log10(P1_I816), name=plotTitle, line=dict(color=currentColor)))
            line_figs['V816_T'].add_trace(
                go.Scatter(x=ts_V_816_time, y=ts_V_816_data, name=plotTitle, line=dict(color=currentColor)))
            line_figs['V816_F'].add_trace(
                go.Scatter(x=f_V816, y=20 * np.log10(P1_V816), name=plotTitle, line=dict(color=currentColor)))

            # Adiciona dados para os gráficos de barra
            legend_entries.append(plotTitle)
            bar_labels.append(plotTitle)
            bar_data_I_Pico.append(data['analise'][0, 0]['Picos'][0, 0]['I800_A_max'][0, 0])
            bar_data_V_Pico.append(data['analise'][0, 0]['Picos'][0, 0]['V800_A_max'][0, 0])
            bar_data_I_h3.append(data['analise'][0, 0]['Harmonicos_h3'][0, 0]['I800_h3_A'][0, 0])
            bar_data_V_h3.append(data['analise'][0, 0]['Harmonicos_h3'][0, 0]['V800_h3_A'][0, 0])

        except Exception as e:
            # Mostra um erro amigável no app
            st.error(f"Erro ao processar o arquivo '{matFile}': {e}. Verifique a estrutura do .mat.")
            keys = [k for k in data.keys() if not k.startswith('__')]
            st.warning(f"Claves encontradas no arquivo: {keys}")
            continue

    # --- 3. Cria as 4 figuras de barra ---
    bar_figs = {}
    if bar_labels:  # Só cria se os dados foram lidos
        df_barras = pd.DataFrame({
            'Caso': bar_labels,
            'I_Pico (A)': bar_data_I_Pico,
            'I_H3 (A)': bar_data_I_h3,
            'V_Pico (V)': bar_data_V_Pico,
            'V_H3 (V)': bar_data_V_h3,
            'Cor': colors[:numFiles]
        })

        bar_figs['Bar_I_Pico'] = go.Figure(
            go.Bar(x=df_barras['Caso'], y=df_barras['I_Pico (A)'], marker_color=df_barras['Cor'],
                   name='Pico de Corrente'))
        bar_figs['Bar_I_Pico'].update_layout(title='Pico de Corrente (I_800, Fase A)',
                                             yaxis_title='Corrente de Pico (A)', xaxis_tickangle=-45)

        bar_figs['Bar_I_h3'] = go.Figure(
            go.Bar(x=df_barras['Caso'], y=df_barras['I_H3 (A)'], marker_color=df_barras['Cor'], name='H3 Corrente'))
        bar_figs['Bar_I_h3'].update_layout(title='3ª Harmônica - Corrente (I_800, Fase A)',
                                           yaxis_title=f'Amplitude H3 (A) @ {f_h3} Hz', xaxis_tickangle=-45)

        bar_figs['Bar_V_Pico'] = go.Figure(
            go.Bar(x=df_barras['Caso'], y=df_barras['V_Pico (V)'], marker_color=df_barras['Cor'],
                   name='Pico de Tensão'))
        bar_figs['Bar_V_Pico'].update_layout(title='Pico de Tensão (V_800, Fase A)', yaxis_title='Tensão de Pico (V)',
                                             xaxis_tickangle=-45)

        bar_figs['Bar_V_h3'] = go.Figure(
            go.Bar(x=df_barras['Caso'], y=df_barras['V_H3 (V)'], marker_color=df_barras['Cor'], name='H3 Tensão'))
        bar_figs['Bar_V_h3'].update_layout(title='3ª Harmônica - Tensão (V_800, Fase A)',
                                           yaxis_title=f'Amplitude H3 (V) @ {f_h3} Hz', xaxis_tickangle=-45)

    # Retorna todas as figuras
    return line_figs, bar_figs


# ===================== Interface Principal do Streamlit =====================

# --- 1. Upload de Arquivos ---
# Substitui o tkinter.filedialog
uploaded_files = st.file_uploader(
    "Selecione os arquivos .mat da sua análise (da pasta 'resultados_faltas_com_fft')",
    accept_multiple_files=True,
    type=['.mat']
)

if uploaded_files:
    # --- 2. Processamento ---
    # Mostra um "spinner" enquanto os dados são processados
    with st.spinner(f'Processando {len(uploaded_files)} arquivos... (Isso pode levar um momento na primeira vez)'):
        # Chama a função cacheada
        line_figs, bar_figs = processar_arquivos(tuple(uploaded_files))

    st.success(f'{len(uploaded_files)} arquivos processados com sucesso!')
    st.markdown("---")

    # --- 3. Exibição dos Gráficos em Abas ---
    tab_metrics, tab_800, tab_816 = st.tabs([
        "📊 Comparação de Métricas (Gráficos de Barra)",
        "📈 Formas de Onda (Ponto 800)",
        "📉 Formas de Onda (Ponto 816)"
    ])

    # Aba de Métricas (Gráficos de Barra)
    with tab_metrics:
        if not bar_figs:
            st.error("Nenhum dado válido foi extraído para gerar os gráficos de barra.")
        else:
            st.plotly_chart(bar_figs['Bar_I_Pico'], use_container_width=True)
            st.plotly_chart(bar_figs['Bar_I_h3'], use_container_width=True)
            st.plotly_chart(bar_figs['Bar_V_Pico'], use_container_width=True)
            st.plotly_chart(bar_figs['Bar_V_h3'], use_container_width=True)

    # Aba Ponto 800 (Formas de Onda)
    with tab_800:
        st.plotly_chart(line_figs['I800_T'], use_container_width=True)
        st.plotly_chart(line_figs['I800_F'], use_container_width=True)
        st.plotly_chart(line_figs['V800_T'], use_container_width=True)
        st.plotly_chart(line_figs['V800_F'], use_container_width=True)

    # Aba Ponto 816 (Formas de Onda)
    with tab_816:
        st.plotly_chart(line_figs['I816_T'], use_container_width=True)
        st.plotly_chart(line_figs['I816_F'], use_container_width=True)
        st.plotly_chart(line_figs['V816_T'], use_container_width=True)
        st.plotly_chart(line_figs['V816_F'], use_container_width=True)

    # --- 4. Funcionalidade de Download ---
    # Coloca o botão de download na barra lateral
    st.sidebar.header("Download dos Gráficos")

    if st.sidebar.button("Preparar Pacote de Download (.zip)"):
        with st.spinner("Gerando arquivos .png e .html..."):

            # Cria um arquivo zip em memória
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
                # Junta os dicionários de figuras
                all_figs = {**line_figs, **bar_figs}

                for name, fig in all_figs.items():
                    fname_base = FNAMES_MAP.get(name, name)

                    # Salva PNG em memória e adiciona ao zip
                    png_bytes = fig.to_image(format="png", width=1200, height=700)
                    zf.writestr(f"{fname_base}.png", png_bytes)

                    # Salva HTML em memória e adiciona ao zip
                    html_str = fig.to_html()
                    zf.writestr(f"{fname_base}.html", html_str)

            # Oferece o zip para download
            st.sidebar.download_button(
                label="Clique para Baixar o ZIP",
                data=zip_buffer.getvalue(),
                file_name="analise_comparativa_plots.zip",
                mime="application/zip"
            )
            st.sidebar.success("Pacote ZIP pronto!")

else:
    st.info("Por favor, carregue um ou mais arquivos .mat para iniciar a análise.")