import streamlit as st
import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# Configuração da página "Wide" para gráficos grandes
st.set_page_config(page_title="Análise Harmônica Detalhada", layout="wide")

st.title("📊 Análise Harmônica por Ponto de Medição")
st.markdown("""
Esta ferramenta agrupa Tensão e Corrente por ponto de medição (ex: 800, 818).
Os gráficos são empilhados para máxima visibilidade das fases.
""")

# --- Função Auxiliar: FFT ---
def calcular_fft(time, signal):
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, dt)
    
    pos_mask = fft_freq >= 0
    return fft_freq[pos_mask], 2.0/n * np.abs(fft_vals[pos_mask]), fs

# --- Função de Plotagem Genérica ---
def plotar_fases_empilhadas(titulo_sinal, dados_cache, var_name, max_freq, min_mag):
    """Gera 3 gráficos (A, B, C) um embaixo do outro para a variável especificada."""
    
    st.markdown(f"### 📈 {titulo_sinal} ({var_name})")
    
    fases_info = [('Fase A', 0, 'red'), ('Fase B', 1, 'blue'), ('Fase C', 2, 'green')]
    
    # Itera sobre as 3 fases (cria 3 gráficos verticais)
    for fase_nome, idx_fase, cor in fases_info:
        fig = go.Figure()
        tem_dados = False
        
        for fname, content in dados_cache.items():
            if hasattr(content['ts'], var_name):
                signal_obj = getattr(content['ts'], var_name)
                
                # Extração segura Time/Data
                try:
                    t = signal_obj.Time
                    y_raw = signal_obj.Data
                except:
                    t = signal_obj.time
                    y_raw = signal_obj.signals.values
                
                # Verifica dimensão (Trifásico ou Monofásico)
                if y_raw.ndim > 1 and y_raw.shape[1] > idx_fase:
                    y_sig = y_raw[:, idx_fase]
                else:
                    # Se pedir Fase B ou C mas o sinal for mono, pula ou repete (optei por pular)
                    if idx_fase > 0: continue 
                    y_sig = y_raw.flatten()

                # FFT
                freqs, mags, _ = calcular_fft(t, y_sig)
                
                # Filtro Harmônico (> 90Hz) e Visual (< max_freq)
                mask = (freqs >= 90) & (freqs <= max_freq)
                x_plot = freqs[mask]
                y_plot = mags[mask]
                
                # Filtro de Ruído visual
                mask_noise = y_plot > min_mag
                
                if len(x_plot[mask_noise]) > 0:
                    fig.add_trace(go.Bar(
                        x=x_plot[mask_noise],
                        y=y_plot[mask_noise],
                        name=f"{fname}",
                        opacity=0.7,
                        marker_color=cor if len(dados_cache) == 1 else None # Cor fixa se for 1 arquivo, variada se forem vários
                    ))
                    tem_dados = True

        if tem_dados:
            fig.update_layout(
                title=f"{fase_nome}",
                xaxis_title="Frequência (Hz)",
                yaxis_title="Magnitude",
                height=400, # Aumentado para melhor visualização vertical
                margin=dict(l=20, r=20, t=40, b=20),
                legend=dict(orientation="h", y=1.1)
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Sidebar ---
st.sidebar.header("Carregar Dados")
uploaded_files = st.sidebar.file_uploader("Arquivos .mat", type=["mat"], accept_multiple_files=True)

if uploaded_files:
    # 1. Carregar e Cachear Dados
    data_cache = {}
    all_vars = set()
    
    for f in uploaded_files:
        try:
            mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
            if 'ts' in mat:
                data_cache[f.name] = {'ts': mat['ts']}
                # Coletar todas as variáveis 'ts_'
                vars_file = [v for v in dir(mat['ts']) if v.startswith('ts_')]
                all_vars.update(vars_file)
        except Exception as e:
            st.error(f"Erro em {f.name}: {e}")

    # 2. Identificar Pontos de Medição Únicos (ex: 800, 818, T2F)
    # Regex procura padrão após ts_V_ ou ts_I_
    pontos_encontrados = set()
    for v in all_vars:
        match = re.search(r'ts_[VI]_(.+)', v)
        if match:
            pontos_encontrados.add(match.group(1))
            
    lista_pontos = sorted(list(pontos_encontrados))

    if not lista_pontos:
        st.warning("Nenhuma variável de Tensão (V) ou Corrente (I) encontrada no padrão 'ts_V_Nome' ou 'ts_I_Nome'.")
    else:
        # 3. Comandos Globais (Acima dos Gráficos)
        st.markdown("---")
        c1, c2 = st.columns(2)
        max_freq = c1.slider("🔍 Zoom Frequência (Máx Hz)", 120, 10000, 2000, step=100)
        min_mag = c2.number_input("🧹 Filtro de Magnitude (Mínima)", value=0.001, format="%.4f", step=0.001)
        st.markdown("---")

        # 4. Criar Abas por Ponto
        tabs = st.tabs([f"📍 Ponto {p}" for p in lista_pontos])
        
        for i, ponto in enumerate(lista_pontos):
            with tabs[i]:
                # Dentro da aba do ponto, verificar se existem V e I
                var_v = f"ts_V_{ponto}"
                var_i = f"ts_I_{ponto}"
                
                tem_v = any(hasattr(d['ts'], var_v) for d in data_cache.values())
                tem_i = any(hasattr(d['ts'], var_i) for d in data_cache.values())
                
                # Seção de Tensão
                if tem_v:
                    st.subheader(f"Tensão no Ponto {ponto}")
                    plotar_fases_empilhadas(f"Tensão {ponto}", data_cache, var_v, max_freq, min_mag)
                    st.divider()
                
                # Seção de Corrente
                if tem_i:
                    st.subheader(f"Corrente no Ponto {ponto}")
                    plotar_fases_empilhadas(f"Corrente {ponto}", data_cache, var_i, max_freq, min_mag)

else:
    st.info("Aguardando upload dos arquivos...")