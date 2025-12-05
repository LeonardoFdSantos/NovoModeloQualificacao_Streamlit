import streamlit as st
import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# Configuração visual
st.set_page_config(page_title="Análise Harmônica - Final", layout="wide")

st.title("📊 Análise Harmônica Detalhada (Empilhada)")
st.markdown("""
Visualização sequencial das fases (A, B, C) com **Janelamento de Harmônicas** e 
**Indicadores de THD** para identificação rápida de problemas.
""")

# --- Funções de Processamento ---
def calcular_fft(time, signal):
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, dt)
    
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    mags = 2.0/n * np.abs(fft_vals[pos_mask])
    
    return freqs, mags, fs

def extrair_picos_harmonicos(freqs, mags, frequencia_fundamental=60, max_ordem=15):
    """Extrai picos exatos nas janelas das harmônicas."""
    dados_harmonicas = {} 
    janela_busca = 5.0 # Hz
    
    for ordem in range(2, max_ordem + 1): 
        freq_alvo = frequencia_fundamental * ordem
        mask_janela = (freqs >= freq_alvo - janela_busca) & (freqs <= freq_alvo + janela_busca)
        
        if np.any(mask_janela):
            pico_mag = np.max(mags[mask_janela])
            dados_harmonicas[ordem] = pico_mag
        else:
            dados_harmonicas[ordem] = 0.0
            
    return dados_harmonicas

def calcular_thd(freqs, mags):
    """Calcula THD considerando a fundamental em 60Hz."""
    idx_60 = (np.abs(freqs - 60)).argmin()
    mag_fund = mags[idx_60]
    if mag_fund == 0: return 0
    
    harm_sq_sum = 0
    for h in range(2, 50): # THD até 50ª ordem
        target = 60 * h
        if target > freqs[-1]: break
        idx = (np.abs(freqs - target)).argmin()
        harm_sq_sum += mags[idx]**2
        
    return (np.sqrt(harm_sq_sum) / mag_fund) * 100

def ordenar_pontos(lista_encontrada):
    ordem_desejada = ['800', 'T2F', '818', '820', '822']
    def get_sort_key(ponto):
        if ponto in ordem_desejada: return ordem_desejada.index(ponto)
        return 999 
    return sorted(lista_encontrada, key=get_sort_key)

# --- Componente Visual Principal ---
def plotar_janelamento_com_thd(titulo, dados_cache, var_name):
    st.markdown(f"### ⚡ {titulo}")
    
    # Configuração das Fases (Uma embaixo da outra)
    fases_cfg = [('Fase A', 0), ('Fase B', 1), ('Fase C', 2)]
    
    for nome_fase, idx_fase in fases_cfg:
        st.markdown(f"#### {nome_fase}")
        
        # 1. Coleta de Dados e Cálculo de THD
        fig = go.Figure()
        thd_results = {}
        tem_dados = False
        
        for fname, content in dados_cache.items():
            if hasattr(content['ts'], var_name):
                signal_obj = getattr(content['ts'], var_name)
                try:
                    t = signal_obj.Time
                    y_raw = signal_obj.Data
                except:
                    t = signal_obj.time
                    y_raw = signal_obj.signals.values
                
                # Seleção da Fase
                if y_raw.ndim > 1 and y_raw.shape[1] > idx_fase:
                    y_sig = y_raw[:, idx_fase]
                else:
                    if idx_fase > 0: continue
                    y_sig = y_raw.flatten()
                
                # Processamento FFT e THD
                freqs, mags, _ = calcular_fft(t, y_sig)
                thd = calcular_thd(freqs, mags)
                thd_results[fname] = thd
                
                # Dados para o Gráfico de Barras (Janelamento)
                picos = extrair_picos_harmonicos(freqs, mags, max_ordem=13)
                ordens = [f"{o}ª" for o in picos.keys()]
                valores = list(picos.values())
                
                # Adiciona traço ao gráfico
                fig.add_trace(go.Bar(
                    x=ordens, 
                    y=valores,
                    name=f"{fname}",
                    text=[f"{v:.3f}" for v in valores],
                    textposition='auto'
                ))
                tem_dados = True
        
        # 2. Exibição dos Indicadores de THD (Estilo Métricas)
        if thd_results:
            cols = st.columns(len(thd_results))
            for i, (arq, val_thd) in enumerate(thd_results.items()):
                # Lógica de cor para o THD
                delta_lbl = "Normal"
                delta_col = "normal"
                if val_thd > 5.0: 
                    delta_lbl = "Alto (>5%)"
                    delta_col = "inverse"
                
                cols[i].metric(
                    label=f"THD - {arq}",
                    value=f"{val_thd:.2f}%",
                    delta=delta_lbl,
                    delta_color=delta_col
                )

        # 3. Exibição do Gráfico de Janelamento
        if tem_dados:
            fig.update_layout(
                title=f"Harmônicas Individuais ({nome_fase})",
                xaxis_title="Ordem Harmônica",
                yaxis_title="Magnitude",
                barmode='group', # Barras lado a lado para comparação
                height=350,
                legend=dict(orientation="h", y=-0.3),
                margin=dict(l=20, r=20, t=30, b=20)
            )
            # KEY ÚNICO para evitar erros
            st.plotly_chart(fig, use_container_width=True, key=f"janela_{var_name}_{nome_fase}")
            st.divider() # Linha separadora entre fases
        else:
            st.info(f"Sem dados para {nome_fase}")

# --- Gráfico de Contexto (Opcional, no Expander) ---
def plotar_espectro_geral(titulo, dados_cache, var_name, max_freq):
    fases = [('Fase A', 0), ('Fase B', 1), ('Fase C', 2)]
    cols = st.columns(3)
    
    for i, (f_nome, idx) in enumerate(fases):
        with cols[i]:
            fig = go.Figure()
            for fname, content in dados_cache.items():
                if hasattr(content['ts'], var_name):
                    sig = getattr(content['ts'], var_name)
                    try: d = sig.Data 
                    except: d = sig.signals.values
                    try: t = sig.Time
                    except: t = sig.time
                    
                    if d.ndim > 1 and d.shape[1] > idx: y = d[:, idx]
                    else: 
                        if idx > 0: continue
                        y = d.flatten()
                        
                    freqs, mags, _ = calcular_fft(t, y)
                    mask = (freqs >= 90) & (freqs <= max_freq)
                    if np.any(mask):
                        fig.add_trace(go.Scatter(x=freqs[mask], y=mags[mask], name=fname))
            
            fig.update_layout(
                title=f_nome, height=200, showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            st.plotly_chart(fig, use_container_width=True, key=f"espectro_{var_name}_{f_nome}")

# --- App Principal ---
st.sidebar.header("1. Upload")
uploaded_files = st.sidebar.file_uploader("Arquivos .mat", type=["mat"], accept_multiple_files=True)
st.sidebar.divider()
st.sidebar.header("2. Configuração")
max_freq_view = st.sidebar.slider("Zoom Espectro Geral (Hz)", 120, 5000, 1200)

if uploaded_files:
    data_cache = {}
    all_vars = set()
    for f in uploaded_files:
        try:
            mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
            if 'ts' in mat:
                data_cache[f.name] = {'ts': mat['ts']}
                all_vars.update([v for v in dir(mat['ts']) if v.startswith('ts_')])
        except: pass

    pontos = set()
    for v in all_vars:
        match = re.search(r'ts_[VI]_(.+)', v)
        if match: pontos.add(match.group(1))
    lista_ordenada = ordenar_pontos(list(pontos))

    # Criação das Abas por Ponto
    tabs = st.tabs([f"📍 {p}" for p in lista_ordenada])

    for i, ponto in enumerate(lista_ordenada):
        with tabs[i]:
            var_i = f"ts_I_{ponto}"
            var_v = f"ts_V_{ponto}"
            
            # --- CORRENTE (PRIMEIRO) ---
            if any(hasattr(d['ts'], var_i) for d in data_cache.values()):
                st.subheader(f"Corrente: {ponto}")
                plotar_janelamento_com_thd(f"Corrente {ponto}", data_cache, var_i)
                
                with st.expander("Ver Espectro Completo (Ruído de Fundo)"):
                    plotar_espectro_geral(f"I_{ponto}", data_cache, var_i, max_freq_view)
                st.markdown("---")

            # --- TENSÃO (SEGUNDO) ---
            if any(hasattr(d['ts'], var_v) for d in data_cache.values()):
                st.subheader(f"Tensão: {ponto}")
                plotar_janelamento_com_thd(f"Tensão {ponto}", data_cache, var_v)
                
                with st.expander("Ver Espectro Completo (Ruído de Fundo)"):
                    plotar_espectro_geral(f"V_{ponto}", data_cache, var_v, max_freq_view)

else:
    st.info("Aguardando upload dos arquivos...")