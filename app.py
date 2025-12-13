import os
import numpy as np
import pandas as pd
import streamlit as st
import pyqtgraph as pg
from scipy.io import loadmat
import matplotlib.pyplot as plt

# Função para carregar e extrair os dados de .mat
def try_extract_ts(mat, ts_key):
    if 'ts' not in mat:
        return None, None
    ts_struct = mat['ts']
    try:
        root = ts_struct[0, 0]
    except Exception:
        return None, None

    if ts_key not in root.dtype.names:
        return None, None

    entry = root[ts_key][0, 0]  # struct Time/Data
    if 'Time' not in entry.dtype.names or 'Data' not in entry.dtype.names:
        return None, None

    t = entry['Time'].squeeze()
    x = entry['Data']
    x = np.array(x)
    x = np.squeeze(x)
    return t, x

# Função de Transformada de Clarke
def clarke_transform(a, b, c, mode="power"):
    if mode == "amp":
        k = 2 / 3
    else:
        k = np.sqrt(2 / 3)
    alpha = k * (a - 0.5 * b - 0.5 * c)
    beta = k * ((np.sqrt(3) / 2) * b - (np.sqrt(3) / 2) * c)
    return alpha, beta

# Função para exibir gráficos no Streamlit
def plot_abc(t, a, b, c, alpha=None, beta=None):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    ax[0].plot(t, a, label='Fase A', color='red')
    ax[0].plot(t, b, label='Fase B', color='green')
    ax[0].plot(t, c, label='Fase C', color='blue')
    ax[0].set_title('Sinais ABC no Tempo')
    ax[0].set_xlabel('Tempo (s)')
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()

    if alpha is not None and beta is not None:
        ax[1].plot(t, alpha, label='Alpha (Clarke)', color='orange')
        ax[1].plot(t, beta, label='Beta (Clarke)', color='cyan')
        ax[1].set_title('Transformada de Clarke αβ no Tempo')
        ax[1].set_xlabel('Tempo (s)')
        ax[1].set_ylabel('Amplitude')
        ax[1].legend()

    st.pyplot(fig)

# Função principal do Streamlit
def main():
    st.title('Visualizador de Simulação IEEE34')

    # Seleção de pasta
    folder_path = st.sidebar.text_input("Caminho da pasta com resultados .mat")
    if folder_path and os.path.exists(folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]
        files.sort()

        file_name = st.sidebar.selectbox("Escolha o arquivo .mat", files)

        if file_name:
            mat_path = os.path.join(folder_path, file_name)
            mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            
            # Mostra metadados
            st.sidebar.subheader(f"Metadados do arquivo: {file_name}")
            st.sidebar.write(f"Caminho: {mat_path}")

            # Pega o ponto de medição para mostrar no gráfico
            point = st.sidebar.selectbox("Selecione o ponto de medição", ['I_800', 'V_800', 'I_T2F', 'V_T2F', 'I_818', 'V_818', 'I_820', 'V_820', 'I_822', 'V_822'])
            ts_key = f"ts_{point}"
            t, x = try_extract_ts(mat, ts_key)

            if t is not None and x is not None:
                # Limpar offset (se necessário)
                remove_mean = st.sidebar.checkbox("Remover Offset (subtrair média)", value=True)
                if remove_mean:
                    x = x - np.mean(x, axis=0)

                # Mostrar gráficos ABC
                if x.ndim == 1:
                    # Se for monofásico (1D), só mostra A
                    a = x
                    b = np.zeros_like(a)
                    c = np.zeros_like(a)
                    alpha, beta = None, None
                else:
                    # Se for trifásico
                    a, b, c = x[:, 0], x[:, 1], x[:, 2]
                    alpha, beta = clarke_transform(a, b, c, mode="power")

                # Plota ABC e αβ
                plot_abc(t, a, b, c, alpha, beta)
                
                # Exibe parâmetros básicos
                st.write(f"Arquivo: {file_name}")
                st.write(f"Tipo de Falta: {file_name.split('__')[-1].replace('_', ' ')}")
                st.write(f"Tempo de Simulação: {t[-1]:.2f} s")
            else:
                st.warning("Não foi possível extrair os dados do arquivo.")

# Executar o app Streamlit
if __name__ == '__main__':
    main()
