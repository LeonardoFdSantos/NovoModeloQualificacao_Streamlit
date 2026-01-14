import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.io as sio
import h5py
import tempfile
import os
from scipy.fft import rfft, rfftfreq

# ==============================================================================
# 1. CONSTANTES E CONFIGURAÇÕES
# ==============================================================================
BARRAS_DISPONIVEIS = ["800", "818", "820", "822", "T2F1", "T2F"]
FREQ_SISTEMA = 60.0
INSTANTE_FALTA = 0.5 / 3  # ~0.1667 s

st.set_page_config(page_title="Análise Tese - Gerador de Figuras", layout="wide")

# ==============================================================================
# 2. MOTOR MATEMÁTICO (DSP)
# ==============================================================================

def calculate_rms(signal):
    """RMS escalar de um vetor."""
    if len(signal) == 0: return 0.0
    return np.sqrt(np.mean(np.square(signal)))

def sliding_rms(signal, fs, window_cycles=1):
    """RMS deslizante para gráficos no tempo."""
    window_size = int((fs / FREQ_SISTEMA) * window_cycles)
    if window_size < 1: window_size = 1
    # Usando convolução para performance (mais rápido que pandas rolling em loops grandes)
    sq = signal ** 2
    window = np.ones(window_size) / window_size
    # mode='same' mantém o tamanho, mas atrasa a fase. Para plot visual é aceitável.
    rms = np.sqrt(np.convolve(sq, window, mode='same'))
    return rms

def get_phasor(signal, fs):
    """Extrai fasor fundamental via DFT."""
    N = len(signal)
    if N == 0: return 0j
    t = np.arange(N) / fs
    kernel = np.exp(-1j * 2 * np.pi * FREQ_SISTEMA * t)
    X = (2.0 / N) * np.sum(signal * kernel)
    return X

def get_sym_components(Va, Vb, Vc):
    """Retorna V0, V1, V2 (Zero, Positiva, Negativa)."""
    a = np.exp(1j * 2 * np.pi / 3)
    V0 = (Va + Vb + Vc) / 3.0
    V1 = (Va + a * Vb + a**2 * Vc) / 3.0
    V2 = (Va + a**2 * Vb + a * Vc) / 3.0
    return V0, V1, V2

def calculate_fft_thd(signal, fs, max_h=40):
    """Retorna V1, V3, THD% e Espectro."""
    N = len(signal)
    if N == 0: return 0, 0, 0, [], []
    
    yf = rfft(signal)
    xf = rfftfreq(N, 1 / fs)
    mag = 2.0 / N * np.abs(yf)
    
    # Índices mais próximos
    idx_fund = (np.abs(xf - FREQ_SISTEMA)).argmin()
    idx_3rd = (np.abs(xf - 3 * FREQ_SISTEMA)).argmin()
    
    V1 = mag[idx_fund]
    V3 = mag[idx_3rd]
    
    # THD
    harmonics_sq = 0
    for h in range(2, max_h + 1):
        idx = (np.abs(xf - h * FREQ_SISTEMA)).argmin()
        if idx < len(mag):
            harmonics_sq += mag[idx]**2
            
    THD = (np.sqrt(harmonics_sq) / (V1 + 1e-9)) * 100
    return V1, V3, THD, xf, mag

# ==============================================================================
# 3. LEITURA DE ARQUIVOS .MAT
# ==============================================================================

def garantir_shape_3fases(arr):
    """Garante que o array seja (N, 3)."""
    arr = np.array(arr)
    if arr.size == 0: return np.zeros((0, 3))
    if arr.ndim == 1: return np.column_stack([arr, arr, arr])
    if arr.shape[0] == 3 and arr.shape[1] > 3: return arr.T
    if arr.ndim == 2 and arr.shape[1] < 3:
        cols_faltantes = 3 - arr.shape[1]
        return np.hstack([arr, np.zeros((arr.shape[0], cols_faltantes))])
    return arr

def ler_arquivo_mat_para_df(uploaded_file):
    """Lê .mat e retorna DataFrame padronizado."""
    # Salvar temporariamente
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mat')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    dados_brutos = {}
    t = None
    
    try:
        # 1. Tenta HDF5
        with h5py.File(tfile.name, 'r') as f:
            keys = list(f.keys())
            for k in keys:
                if 'time' in k or k == 't':
                    t = np.array(f[k]).flatten()
                    break
            
            for barra in BARRAS_DISPONIVEIS:
                # Tenta variações de nome
                v_key = next((k for k in keys if f"V_{barra}" in k and "raw" not in k), None)
                if not v_key: v_key = next((k for k in keys if f"V_{barra}" in k), None)
                
                i_key = next((k for k in keys if f"I_{barra}" in k and "raw" not in k), None)
                if not i_key: i_key = next((k for k in keys if f"I_{barra}" in k), None)

                if v_key and i_key:
                    V = garantir_shape_3fases(f[v_key])
                    I = garantir_shape_3fases(f[i_key])
                    dados_brutos[barra] = {"V": V, "I": I}

    except OSError:
        # 2. Tenta Scipy
        try:
            mat = sio.loadmat(tfile.name)
            keys = list(mat.keys())
            for k in keys:
                if 'time' in k or k == 't':
                    t = mat[k].flatten()
                    break
            
            for barra in BARRAS_DISPONIVEIS:
                v_key = next((k for k in keys if f"V_{barra}" in k), None)
                i_key = next((k for k in keys if f"I_{barra}" in k), None)
                
                if v_key and i_key:
                    V = garantir_shape_3fases(mat[v_key])
                    I = garantir_shape_3fases(mat[i_key])
                    dados_brutos[barra] = {"V": V, "I": I}
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")
            return None, 0
    finally:
        os.remove(tfile.name)

    if t is None or not dados_brutos:
        return None, 0

    # Construção do DataFrame
    # Sincroniza tamanho pelo menor vetor encontrado para evitar erro de dimensão
    min_len = len(t)
    for b in dados_brutos:
        min_len = min(min_len, len(dados_brutos[b]['V']), len(dados_brutos[b]['I']))
    
    # Se o menor tamanho for muito pequeno (lixo), aborta
    if min_len < 10: return None, 0

    df_dict = {'t': t[:min_len]}
    fs = 1.0 / (t[1] - t[0]) if len(t) > 1 else 60*256

    for barra, d in dados_brutos.items():
        # Só adiciona se tiver tamanho compatível
        if len(d['V']) >= min_len and len(d['I']) >= min_len:
            v = d['V'][:min_len]
            i = d['I'][:min_len]
            
            df_dict[f"V_{barra}_A"] = v[:, 0]; df_dict[f"V_{barra}_B"] = v[:, 1]; df_dict[f"V_{barra}_C"] = v[:, 2]
            df_dict[f"I_{barra}_A"] = i[:, 0]; df_dict[f"I_{barra}_B"] = i[:, 1]; df_dict[f"I_{barra}_C"] = i[:, 2]

    return pd.DataFrame(df_dict), fs

# ==============================================================================
# 4. INTERFACE E LÓGICA DE GRÁFICOS
# ==============================================================================

# --- Sidebar: Upload ---
st.sidebar.header("1. Upload de Arquivos .MAT")
uploaded_files = st.sidebar.file_uploader("Carregue arquivos para comparar", accept_multiple_files=True, type=['mat'])
data_store = []

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        # Auto-detectar metadados do nome
        fname = uploaded_file.name.lower()
        
        d_topo = 1 if "t2f" in fname else 0
        d_reg = 1 if "_sr_" in fname or "sem_reg" in fname else 0
        d_terra = 1 if "sem_terra" in fname else 0
        
        # Falta
        d_falta = 0 # Pleno
        if "falta_a" in fname and "falta_ab" not in fname: d_falta = 1 # A-Terra
        elif "falta_abc" in fname: d_falta = 2
        elif "falta_ab" in fname: d_falta = 3
        elif "falta_bc" in fname: d_falta = 4
        
        st.sidebar.markdown(f"**Arq {i+1}: {uploaded_file.name}**")
        c1, c2 = st.sidebar.columns(2)
        topo = c1.selectbox(f"Topol #{i+1}", ["MRN", "T2F"], index=d_topo, key=f"t{i}")
        reg = c2.selectbox(f"Regul #{i+1}", ["Com Reg", "Sem Reg"], index=d_reg, key=f"r{i}")
        c3, c4 = st.sidebar.columns(2)
        terra = c3.selectbox(f"GND #{i+1}", ["Com Terra", "Sem Terra"], index=d_terra, key=f"g{i}")
        falta = c4.selectbox(f"Falta #{i+1}", ["Pleno", "A-Terra", "ABC", "AB", "BC"], index=d_falta, key=f"f{i}")
        
        # Leitura
        uploaded_file.seek(0)
        df, fs = ler_arquivo_mat_para_df(uploaded_file)
        
        if df is not None:
            data_store.append({
                "label": f"{topo} | {reg} | {terra} | {falta}",
                "df": df,
                "fs": fs,
                "meta": {"topo": topo, "reg": reg, "terra": terra, "falta": falta}
            })
        else:
            st.sidebar.error(f"Erro ao ler {uploaded_file.name}")

# --- Sidebar: Configuração ---
st.sidebar.markdown("---")
st.sidebar.header("2. Seletor de Figuras")
analise_tipo = st.sidebar.selectbox("Escolha a Figura:", [
    "4.4 - Perfil de Tensão (Regime)",
    "4.5 - Desequilíbrio V2/V1 (Regime)",
    "4.6 - Tensão RMS vs Tempo",
    "4.7 - Corrente RMS vs Tempo",
    "4.8 - Componentes Simétricas (Falta)",
    "4.9/10 - FFT e Harmônicas",
    "4.11 - Razão V3/V1 (%)",
    "4.12 - Corrente Máxima de Falta",
    "4.13 - Razão I0/I1 (%)"
])

barra_sel = st.sidebar.selectbox("Barra de Interesse", BARRAS_DISPONIVEIS)

t_max = 0.5
if data_store: t_max = data_store[0]["df"]['t'].iloc[-1]
janela = st.sidebar.slider("Janela de Análise (s)", 0.0, float(t_max), (0.2, 0.4), step=0.01)

# --- ÁREA PRINCIPAL ---
st.title("Gerador de Figuras - Tese")

if not data_store:
    st.info("👈 Carregue arquivos .mat na barra lateral para começar.")
    st.stop()

# Helper para extrair matrizes V e I de uma barra do DataFrame
def get_matrices(df, bus):
    # Retorna V(N,3) e I(N,3) se existirem
    try:
        Va = df[f"V_{bus}_A"].values; Vb = df[f"V_{bus}_B"].values; Vc = df[f"V_{bus}_C"].values
        Ia = df[f"I_{bus}_A"].values; Ib = df[f"I_{bus}_B"].values; Ic = df[f"I_{bus}_C"].values
        return np.column_stack([Va, Vb, Vc]), np.column_stack([Ia, Ib, Ic])
    except KeyError:
        return None, None

# ==============================================================================
# LÓGICA DE PLOTAGEM
# ==============================================================================

if analise_tipo == "4.4 - Perfil de Tensão (Regime)":
    st.subheader("Figura 4.4 - Perfil de Tensão RMS (Regime Permanente)")
    fig = go.Figure()
    
    for item in data_store:
        df = item["df"]
        # Filtra Janela
        mask = (df['t'] >= janela[0]) & (df['t'] <= janela[1])
        df_win = df.loc[mask]
        
        x_val, y_val = [], []
        for bus in BARRAS_DISPONIVEIS:
            V, _ = get_matrices(df_win, bus)
            if V is not None:
                # RMS médio das 3 fases
                val = np.mean([calculate_rms(V[:, i]) for i in range(3)])
                x_val.append(bus)
                y_val.append(val)
        
        fig.add_trace(go.Scatter(x=x_val, y=y_val, mode='lines+markers', name=item["label"]))
    
    fig.update_layout(xaxis_title="Barra", yaxis_title="Tensão RMS (V)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.5 - Desequilíbrio V2/V1 (Regime)":
    st.subheader("Figura 4.5 - Desequilíbrio de Tensão V2/V1")
    fig = go.Figure()
    
    for item in data_store:
        df, fs = item["df"], item["fs"]
        mask = (df['t'] >= janela[0]) & (df['t'] <= janela[1])
        df_win = df.loc[mask]
        
        x_val, y_val = [], []
        for bus in BARRAS_DISPONIVEIS:
            V, _ = get_matrices(df_win, bus)
            if V is not None:
                Va = get_phasor(V[:,0], fs); Vb = get_phasor(V[:,1], fs); Vc = get_phasor(V[:,2], fs)
                _, V1, V2 = get_sym_components(Va, Vb, Vc)
                ratio = np.abs(V2)/np.abs(V1)*100 if np.abs(V1) > 0 else 0
                x_val.append(bus); y_val.append(ratio)
        
        fig.add_trace(go.Bar(x=x_val, y=y_val, name=item["label"]))
    
    fig.update_layout(barmode='group', xaxis_title="Barra", yaxis_title="V2/V1 (%)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.6 - Tensão RMS vs Tempo":
    st.subheader(f"Figura 4.6 - Tensão RMS no Tempo | Barra {barra_sel}")
    fig = go.Figure()
    
    for item in data_store:
        df, fs = item["df"], item["fs"]
        V, _ = get_matrices(df, barra_sel)
        
        if V is not None:
            # Plota apenas Fase A para não poluir, ou média
            v_rms = sliding_rms(V[:, 0], fs)
            fig.add_trace(go.Scatter(x=df['t'], y=v_rms, name=f"{item['label']} (Fase A)"))
            
    fig.add_vline(x=INSTANTE_FALTA, line_dash="dash", line_color="red", annotation_text="Falta")
    fig.update_layout(xaxis_title="Tempo (s)", yaxis_title="Tensão RMS (V)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.7 - Corrente RMS vs Tempo":
    st.subheader(f"Figura 4.7 - Corrente RMS no Tempo | Barra {barra_sel}")
    fig = go.Figure()
    
    for item in data_store:
        df, fs = item["df"], item["fs"]
        _, I = get_matrices(df, barra_sel)
        
        if I is not None:
            i_rms = sliding_rms(I[:, 0], fs) # Fase A
            fig.add_trace(go.Scatter(x=df['t'], y=i_rms, name=f"{item['label']} (Fase A)"))
            
    fig.add_vline(x=INSTANTE_FALTA, line_dash="dash", line_color="red")
    fig.update_layout(xaxis_title="Tempo (s)", yaxis_title="Corrente RMS (A)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.8 - Componentes Simétricas (Falta)":
    st.subheader(f"Figura 4.8 - Simétricas de Corrente | Barra {barra_sel}")
    fig = go.Figure()
    
    for item in data_store:
        df, fs = item["df"], item["fs"]
        mask = (df['t'] >= janela[0]) & (df['t'] <= janela[1])
        df_win = df.loc[mask]
        
        _, I = get_matrices(df_win, barra_sel)
        if I is not None:
            Ia = get_phasor(I[:,0], fs); Ib = get_phasor(I[:,1], fs); Ic = get_phasor(I[:,2], fs)
            I0, I1, I2 = get_sym_components(Ia, Ib, Ic)
            
            fig.add_trace(go.Bar(x=[item['label']], y=[np.abs(I0)], name='I0', marker_color='gray'))
            fig.add_trace(go.Bar(x=[item['label']], y=[np.abs(I1)], name='I1', marker_color='blue'))
            fig.add_trace(go.Bar(x=[item['label']], y=[np.abs(I2)], name='I2', marker_color='red'))
            
    fig.update_layout(barmode='group', title="Magnitude Componentes (A)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.9/10 - FFT e Harmônicas":
    st.subheader(f"Análise FFT | Barra {barra_sel} (Fase A)")
    fig = go.Figure()
    stats = []
    
    for item in data_store:
        df, fs = item["df"], item["fs"]
        mask = (df['t'] >= janela[0]) & (df['t'] <= janela[1])
        df_win = df.loc[mask]
        
        V, _ = get_matrices(df_win, barra_sel)
        if V is not None:
            sig = V[:, 0] # Fase A
            V1, V3, thd, freqs, mag = calculate_fft_thd(sig, fs)
            
            stats.append({
                "Cenário": item["label"], "V1": f"{V1:.1f}", 
                "V3": f"{V3:.1f}", "THD": f"{thd:.2f}%"
            })
            
            mask_f = freqs <= 600
            fig.add_trace(go.Scatter(x=freqs[mask_f], y=mag[mask_f], name=item["label"]))
            
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame(stats))


elif analise_tipo == "4.11 - Razão V3/V1 (%)":
    st.subheader(f"Figura 4.11 - Distorção V3/V1 (%) | Barra {barra_sel}")
    labels, vals = [], []
    
    for item in data_store:
        df, fs = item["df"], item["fs"]
        mask = (df['t'] >= janela[0]) & (df['t'] <= janela[1])
        df_win = df.loc[mask]
        
        V, _ = get_matrices(df_win, barra_sel)
        if V is not None:
            V1, V3, _, _, _ = calculate_fft_thd(V[:, 0], fs)
            ratio = (V3/V1)*100 if V1 > 0 else 0
            labels.append(item["label"])
            vals.append(ratio)
            
    fig = go.Figure(go.Bar(x=labels, y=vals))
    fig.update_layout(yaxis_title="V3/V1 (%)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.12 - Corrente Máxima de Falta":
    st.subheader(f"Figura 4.12 - Corrente Máxima | Barra {barra_sel}")
    labels, vals = [], []
    
    for item in data_store:
        df, fs = item["df"], item["fs"]
        mask = (df['t'] >= janela[0]) & (df['t'] <= janela[1])
        df_win = df.loc[mask]
        
        _, I = get_matrices(df_win, barra_sel)
        if I is not None:
            # Pega o máximo do RMS deslizante de qualquer fase
            imax = 0
            for i in range(3):
                imax = max(imax, np.max(sliding_rms(I[:, i], fs)))
            labels.append(item["label"])
            vals.append(imax)
            
    fig = go.Figure(go.Bar(x=labels, y=vals))
    fig.update_layout(yaxis_title="Corrente (A)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.13 - Razão I0/I1 (%)":
    st.subheader(f"Figura 4.13 - Razão I0/I1 (%) | Barra {barra_sel}")
    labels, vals = [], []
    
    for item in data_store:
        df, fs = item["df"], item["fs"]
        mask = (df['t'] >= janela[0]) & (df['t'] <= janela[1])
        df_win = df.loc[mask]
        
        _, I = get_matrices(df_win, barra_sel)
        if I is not None:
            Ia = get_phasor(I[:,0], fs); Ib = get_phasor(I[:,1], fs); Ic = get_phasor(I[:,2], fs)
            I0, I1, _ = get_sym_components(Ia, Ib, Ic)
            
            ratio = (np.abs(I0)/np.abs(I1))*100 if np.abs(I1)>0 else 0
            labels.append(item["label"])
            vals.append(ratio)
            
    fig = go.Figure(go.Bar(x=labels, y=vals))
    fig.update_layout(yaxis_title="I0/I1 (%)")
    st.plotly_chart(fig, use_container_width=True)
