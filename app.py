import streamlit as st
import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# Configuração visual
st.set_page_config(page_title="Análise Harmônica Detalhada", layout="wide")

st.title("📊 Análise Detalhada: Janelamento de Harmônicas")
st.markdown("""
Visualização focada nas **magnitudes exatas** de cada ordem harmônica (Janelamento),
facilitando a identificação de problemáticas específicas (ex: excesso de 3ª ou 5ª ordem).
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
    """
    Realiza o 'janelamento': Busca o valor máximo em uma pequena janela 
    ao redor de cada frequência harmônica teórica.
    """
    dados_harmonicas = {} # {Ordem: Magnitude}
    
    janela_busca = 5.0 # Hz (Busca pico em +/- 5Hz da harmônica)
    
    for ordem in range(2, max_ordem + 1): # Começa da 2ª harmônica
        freq_alvo = frequencia_fundamental * ordem
        
        # Máscara para olhar apenas na janela ao redor da harmônica
        mask_janela = (freqs >= freq_alvo - janela_busca) & (freqs <= freq_alvo + janela_busca)
        
        if np.any(mask_janela):
            # Pega o maior pico dentro dessa janela
            pico_mag = np.max(mags[mask_janela])
            dados_harmonicas[ordem] = pico_mag
        else:
            dados_harmonicas[ordem] = 0.0
            
    return dados_harmonicas

def calcular_thd(freqs, mags):
    idx_60 = (np.abs(freqs - 60)).argmin()
    mag_fund = mags[idx_60]
    if mag_fund == 0: return 0
    
    harm_sq_sum = 0
    for h in range(2, 40): # THD até 40ª
        target = 60 * h
        if target > freqs[-1]: break
        # Busca simples pelo mais próximo
        idx = (np.abs(freqs - target)).argmin()
        harm_sq_sum += mags[idx]**2
        
    return (np.sqrt(harm_sq_sum) / mag_fund) * 100

def ordenar_pontos(lista_encontrada):
    ordem_desejada = ['800', 'T2F', '818', '820', '822']
    def get_sort_key(ponto):
        if ponto in ordem_desejada: return ordem_desejada.index(ponto)
        return 999 
    return sorted(lista_encontrada, key=get_sort_key)

# --- Plotagem: Janelamento Comparativo ---
def plotar_janelamento_harmonico(titulo, dados_cache, var_name):
    st.markdown(f"### 🔍 {titulo} - Comparativo por Harmônica")
    
    # Abas internas para Fases (para não poluir tudo num gráfico só)
    tab_a, tab_b, tab_c = st.tabs(["Fase A", "Fase B", "Fase C"])
    
    fases_cfg = [
        (tab_a, 0, 'Fase A'), 
        (tab_b, 1, 'Fase B'), 
        (tab_c, 2, 'Fase C')
    ]
    
    for aba, idx_fase, nome_fase in fases_cfg:
        with aba:
            fig = go.Figure()
            
            # Para cada arquivo carregado...
            for fname, content in dados_cache.items():
                if hasattr(content['ts'], var_name):
                    signal_obj = getattr(content['ts'], var_name)
                    
                    # Extração segura
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
                    
                    # FFT
                    freqs, mags, _ = calcular_fft(t, y_sig)
                    
                    # --- JANELAMENTO: Extrair picos exatos ---
                    picos = extrair_picos_harmonicos(freqs, mags, max_ordem=13) # Até 13ª ordem
                    
                    # Preparar dados para plotagem
                    ordens = [f"{o}ª ({o*60}Hz)" for o in picos.keys()]
                    valores = list(picos.values())
                    
                    # Adiciona barra agrupada para este arquivo
                    fig.add_trace(go.Bar(
                        x=ordens, 
                        y=valores,
                        name=fname,
                        text=[f"{v:.3f}" for v in valores], # Mostra valor no topo
                        textposition='auto'
                    ))
            
            fig.update_layout(
                title=f"Comparação Direta de Harmônicas ({nome_fase})",
                xaxis_title="Ordem Harmônica (Janela)",
                yaxis_title="Magnitude",
                barmode='group', # ISSO FAZ O AGRUPAMENTO LADO A LADO
                height=400,
                legend=dict(orientation="h", y=-0.2),
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

# --- Plotagem: Espectro Geral (Contexto) ---
def plotar_espectro_geral(titulo, dados_cache, var_name, max_freq, min_mag):
    # (Esta função mantém o gráfico de linhas contínuo para contexto geral)
    st.markdown(f"#### 📉 Espectro Geral (Visão Panorâmica) - {titulo}")
    
    fases = [('Fase A', 0), ('Fase B', 1), ('Fase C', 2)]
    cols = st.columns(3)
    
    for i, (f_nome, idx) in enumerate(fases):
        with cols[i]:
            fig = go.Figure()
            thd_msg = []
            
            for fname, content in dados_cache.items():
                if hasattr(content['ts'], var_name):
                    # ... (Lógica de extração igual) ...
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
                    thd = calcular_thd(freqs, mags)
                    thd_msg.append(f"{fname[:10]}..: {thd:.1f}%")
                    
                    mask = (freqs >= 90) & (freqs <= max_freq)
                    mask_noise = mags[mask] > min_mag
                    
                    if np.any(mask_noise):
                        fig.add_trace(go.Scatter( # Scatter/Linha para visão geral
                            x=freqs[mask][mask_noise],
                            y=mags[mask][mask_noise],
                            name=fname,
                            opacity=0.8
                        ))
            
            fig.update_layout(
                title=f"{f_nome}",
                xaxis_title="Hz", 
                yaxis_title="Mag",
                height=250,
                margin=dict(l=0, r=0, t=30, b=0),
                showlegend=False # Legenda polui gráficos pequenos
            )
            st.plotly_chart(fig, use_container_width=True)
            # Mostra THD resumido abaixo
            st.caption("THD: " + " | ".join(thd_msg))

# --- Main App ---
st.sidebar.header("Carregar Arquivos")
uploaded_files = st.sidebar.file_uploader("", type=["mat"], accept_multiple_files=True)
st.sidebar.divider()
max_freq_view = st.sidebar.slider("Zoom Espectro Geral (Hz)", 120, 3000, 1200)

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

    # Ordenação
    pontos = set()
    for v in all_vars:
        match = re.search(r'ts_[VI]_(.+)', v)
        if match: pontos.add(match.group(1))
    lista_ordenada = ordenar_pontos(list(pontos))

    # Abas Principais (Pontos)
    tabs = st.tabs([f"📍 {p}" for p in lista_ordenada])

    for i, ponto in enumerate(lista_ordenada):
        with tabs[i]:
            var_i = f"ts_I_{ponto}"
            var_v = f"ts_V_{ponto}"
            
            # --- 1. SEÇÃO CORRENTE (I) ---
            if any(hasattr(d['ts'], var_i) for d in data_cache.values()):
                st.subheader(f"Corrente: {ponto}")
                # 1.1 - O Novo Gráfico de Janelamento (O Destaque)
                plotar_janelamento_harmonico(f"Corrente {ponto}", data_cache, var_i)
                # 1.2 - O Gráfico Geral (Para contexto)
                with st.expander("Ver Espectro Contínuo (Detalhe de Frequência)"):
                    plotar_espectro_geral(f"I_{ponto}", data_cache, var_i, max_freq_view, 0.001)
                st.divider()

            # --- 2. SEÇÃO TENSÃO (V) ---
            if any(hasattr(d['ts'], var_v) for d in data_cache.values()):
                st.subheader(f"Tensão: {ponto}")
                plotar_janelamento_harmonico(f"Tensão {ponto}", data_cache, var_v)
                with st.expander("Ver Espectro Contínuo (Detalhe de Frequência)"):
                    plotar_espectro_geral(f"V_{ponto}", data_cache, var_v, max_freq_view, 0.001)

else:
    st.info("Aguardando arquivos...")