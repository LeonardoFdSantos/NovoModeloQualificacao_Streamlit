import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.fft import rfft, rfftfreq

# ==============================================================================
# 1. CONSTANTES E CONFIGURAÇÕES
# ==============================================================================
BARRAS_DISPONIVEIS = ["800", "818", "820", "822", "T2F1", "T2F"]
FREQ_SISTEMA = 60.0
INSTANTE_FALTA = 0.5 / 3  # ~0.1667 s

st.set_page_config(page_title="Análise de Proteção - Tese", layout="wide")

# ==============================================================================
# 2. FUNÇÕES MATEMÁTICAS (O MOTOR DE CÁLCULO)
# ==============================================================================

def calculate_rms(signal):
    """RMS escalar de um vetor."""
    return np.sqrt(np.mean(np.square(signal)))

def sliding_rms(signal, fs, window_cycles=1):
    """RMS deslizante para gráficos no tempo."""
    window_size = int((fs / FREQ_SISTEMA) * window_cycles)
    series = pd.Series(signal)
    # Rolling window, shift para centralizar ou alinhar à direita (ajuste conforme necessidade)
    rms = series.rolling(window=window_size).apply(lambda x: np.sqrt(np.mean(x**2)))
    return rms.fillna(0).values

def get_phasor(signal, fs):
    """Extrai fasor fundamental via DFT."""
    N = len(signal)
    if N == 0: return 0j
    t = np.arange(N) / fs
    # Kernel para 60Hz
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
    yf = rfft(signal)
    xf = rfftfreq(N, 1 / fs)
    
    # Normalizar magnitude (Pico)
    mag = 2.0 / N * np.abs(yf)
    
    # Encontrar índices mais próximos
    idx_fund = (np.abs(xf - FREQ_SISTEMA)).argmin()
    idx_3rd = (np.abs(xf - 3 * FREQ_SISTEMA)).argmin()
    
    V1 = mag[idx_fund]
    V3 = mag[idx_3rd]
    
    # THD: Raiz da soma dos quadrados das harmônicas / Fundamental
    # Máscara para harmônicas (exclui DC e Fundamental)
    harmonics_sq = 0
    for h in range(2, max_h + 1):
        idx = (np.abs(xf - h * FREQ_SISTEMA)).argmin()
        if idx < len(mag):
            harmonics_sq += mag[idx]**2
            
    THD = (np.sqrt(harmonics_sq) / V1) * 100 if V1 > 0 else 0
    
    return V1, V3, THD, xf, mag

# ==============================================================================
# 3. INTERFACE LATERAL (INPUTS)
# ==============================================================================

st.sidebar.header("1. Carregamento de Arquivos")
st.sidebar.markdown("Carregue até 4 cenários (CSV processado).")

uploaded_files = st.sidebar.file_uploader("Selecione arquivos CSV", accept_multiple_files=True)
data_store = []

if uploaded_files:
    for i, uploaded_file in enumerate(uploaded_files):
        if i >= 4: break # Limite de 4
        st.sidebar.markdown(f"--- **Arquivo {i+1}: {uploaded_file.name}**")
        
        # Metadados
        col1, col2 = st.sidebar.columns(2)
        topo = col1.selectbox(f"Topologia #{i+1}", ["MRN", "T2F"], key=f"top_{i}")
        reg = col2.selectbox(f"Regulador #{i+1}", ["Com Reg", "Sem Reg"], key=f"reg_{i}")
        terra = col1.selectbox(f"Aterramento #{i+1}", ["Com Terra", "Sem Terra"], key=f"gnd_{i}")
        falta = col2.selectbox(f"Falta #{i+1}", ["Pleno", "A-Terra", "ABC", "AB", "BC"], key=f"flt_{i}")
        
        # Carregar DF
        # Assumindo CSV com colunas: tempo, V_800_A, V_800_B, I_800_A, etc.
        try:
            df = pd.read_csv(uploaded_file)
            # Tentar inferir fs
            t = df.iloc[:, 0].values
            dt = t[1] - t[0]
            fs = 1.0 / dt
            
            data_store.append({
                "name": uploaded_file.name,
                "label": f"{topo} | {reg} | {terra} | {falta}",
                "df": df,
                "fs": fs,
                "meta": {"topo": topo, "reg": reg, "terra": terra, "falta": falta}
            })
        except Exception as e:
            st.error(f"Erro ao ler {uploaded_file.name}: {e}")

st.sidebar.markdown("---")
st.sidebar.header("2. Configuração de Análise")

analise_tipo = st.sidebar.selectbox(
    "Tipo de Figura (Seção 4)",
    [
        "4.4 - Perfil de Tensão (Regime)",
        "4.5 - Desequilíbrio V2/V1 (Regime)",
        "4.6 - Tensão RMS vs Tempo",
        "4.7 - Corrente RMS vs Tempo",
        "4.8 - Componentes Simétricas (Falta)",
        "4.9/10 - FFT e Harmônicas",
        "4.11 - Razão V3/V1 (%)",
        "4.12 - Corrente Máxima de Falta"
    ]
)

barra_selecionada = st.sidebar.selectbox("Barra de Interesse", BARRAS_DISPONIVEIS)
calc_thd = st.sidebar.checkbox("Calcular THD?", value=True)

# Seleção de Janela de Tempo
st.sidebar.subheader("Janela de Tempo")
t_min, t_max = 0.0, 0.5
if data_store:
    t_max = data_store[0]["df"].iloc[-1, 0]

janela = st.sidebar.slider("Intervalo de Análise (s)", 0.0, float(t_max), (0.3, 0.4))

# ==============================================================================
# 4. LÓGICA DE GERAÇÃO DE GRÁFICOS
# ==============================================================================

st.title("Análise de Resultados - Proteção T2F/MRN")

if not data_store:
    st.info("Por favor, carregue os arquivos CSV na barra lateral para começar.")
    st.stop()

# Função auxiliar para pegar dados de uma barra específica do DF
def get_bus_data(df, bus):
    # Procura colunas. Ex: V_800_A ou v_800_a ou Voltage_800_A...
    # Ajuste este padrão conforme o seu CSV
    cols = df.columns
    # Tenta padrão genérico V_{bus}_A
    v_cols = [c for c in cols if f"V_{bus}" in c or f"v_{bus}" in c]
    i_cols = [c for c in cols if f"I_{bus}" in c or f"i_{bus}" in c]
    
    # Ordenar A, B, C
    v_cols.sort() 
    i_cols.sort()
    
    if len(v_cols) < 3 or len(i_cols) < 3:
        return None, None
    
    # Retorna matrizes (N, 3)
    V = df[v_cols].values
    I = df[i_cols].values
    return V, I

# --- LÓGICA POR TIPO DE FIGURA ---

if analise_tipo == "4.4 - Perfil de Tensão (Regime)":
    st.subheader(f"Figura 4.4 - Perfil de Tensão RMS (Janela: {janela})")
    
    fig = go.Figure()
    
    for item in data_store:
        df = item["df"]
        fs = item["fs"]
        # Filtrar janela
        mask = (df.iloc[:,0] >= janela[0]) & (df.iloc[:,0] <= janela[1])
        df_win = df.loc[mask]
        
        y_vals = []
        x_vals = []
        
        for bus in BARRAS_DISPONIVEIS:
            V, _ = get_bus_data(df_win, bus)
            if V is not None:
                # RMS médio das 3 fases
                rms_abc = [calculate_rms(V[:, i]) for i in range(3)]
                y_vals.append(np.mean(rms_abc))
                x_vals.append(bus)
        
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines+markers', name=item["label"]))
        
    fig.update_layout(xaxis_title="Barras", yaxis_title="Tensão RMS (V/pu)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.5 - Desequilíbrio V2/V1 (Regime)":
    st.subheader(f"Figura 4.5 - Desequilíbrio V2/V1 (Janela: {janela})")
    
    fig = go.Figure()
    
    for item in data_store:
        df = item["df"]
        fs = item["fs"]
        mask = (df.iloc[:,0] >= janela[0]) & (df.iloc[:,0] <= janela[1])
        df_win = df.loc[mask]
        
        x_vals = []
        y_vals = []
        
        for bus in BARRAS_DISPONIVEIS:
            V, _ = get_bus_data(df_win, bus)
            if V is not None:
                # Fasores fundamentais
                Va = get_phasor(V[:, 0], fs)
                Vb = get_phasor(V[:, 1], fs)
                Vc = get_phasor(V[:, 2], fs)
                _, V1, V2 = get_sym_components(Va, Vb, Vc)
                
                ratio = (np.abs(V2) / np.abs(V1)) * 100 if np.abs(V1) > 0 else 0
                x_vals.append(bus)
                y_vals.append(ratio)
        
        fig.add_trace(go.Bar(x=x_vals, y=y_vals, name=item["label"]))
        
    fig.update_layout(barmode='group', xaxis_title="Barras", yaxis_title="V2/V1 (%)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.6 - Tensão RMS vs Tempo":
    st.subheader(f"Figura 4.6 - Tensão RMS no Tempo | Barra {barra_selecionada}")
    
    fig = go.Figure()
    
    for item in data_store:
        df = item["df"]
        fs = item["fs"]
        t = df.iloc[:, 0].values
        V, _ = get_bus_data(df, barra_selecionada)
        
        if V is not None:
            # RMS Deslizante Fase A (ou média, aqui plotando Fase A como exemplo)
            v_rms = sliding_rms(V[:, 0], fs)
            
            fig.add_trace(go.Scatter(x=t, y=v_rms, mode='lines', name=f"{item['label']} (Fase A)"))
            
    # Linha de Falta
    fig.add_vline(x=INSTANTE_FALTA, line_width=2, line_dash="dash", line_color="red", annotation_text="Falta")
    fig.update_layout(xaxis_title="Tempo (s)", yaxis_title="Tensão RMS (V)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.7 - Corrente RMS vs Tempo":
    st.subheader(f"Figura 4.7 - Corrente RMS no Tempo | Barra {barra_selecionada}")
    # Igual ao 4.6 mas com Corrente
    fig = go.Figure()
    for item in data_store:
        df = item["df"]
        fs = item["fs"]
        t = df.iloc[:, 0].values
        _, I = get_bus_data(df, barra_selecionada)
        if I is not None:
            i_rms = sliding_rms(I[:, 0], fs)
            fig.add_trace(go.Scatter(x=t, y=i_rms, mode='lines', name=f"{item['label']} (Fase A)"))
            
    fig.add_vline(x=INSTANTE_FALTA, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(xaxis_title="Tempo (s)", yaxis_title="Corrente RMS (A)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.8 - Componentes Simétricas (Falta)":
    st.subheader(f"Figura 4.8 - Componentes Simétricas de Corrente | Barra {barra_selecionada}")
    st.markdown(f"Analisando na janela selecionada: {janela}")
    
    labels = []
    i0_vals, i1_vals, i2_vals = [], [], []
    
    for item in data_store:
        df = item["df"]
        fs = item["fs"]
        mask = (df.iloc[:,0] >= janela[0]) & (df.iloc[:,0] <= janela[1])
        df_win = df.loc[mask]
        
        _, I = get_bus_data(df_win, barra_selecionada)
        if I is not None:
            Ia = get_phasor(I[:, 0], fs)
            Ib = get_phasor(I[:, 1], fs)
            Ic = get_phasor(I[:, 2], fs)
            I0, I1, I2 = get_sym_components(Ia, Ib, Ic)
            
            labels.append(item["label"])
            i0_vals.append(np.abs(I0))
            i1_vals.append(np.abs(I1))
            i2_vals.append(np.abs(I2))
            
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=i0_vals, name='I0 (Zero)'))
    fig.add_trace(go.Bar(x=labels, y=i1_vals, name='I1 (Positiva)'))
    fig.add_trace(go.Bar(x=labels, y=i2_vals, name='I2 (Negativa)'))
    
    fig.update_layout(barmode='group', title="Magnitude Componentes Simétricas (A)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.9/10 - FFT e Harmônicas":
    st.subheader(f"Análise Espectral (FFT) | Barra {barra_selecionada}")
    st.markdown("Visualizando espectro da Fase A na janela selecionada.")
    
    fig = go.Figure()
    stats = []
    
    for item in data_store:
        df = item["df"]
        fs = item["fs"]
        mask = (df.iloc[:,0] >= janela[0]) & (df.iloc[:,0] <= janela[1])
        df_win = df.loc[mask]
        
        V, _ = get_bus_data(df_win, barra_selecionada)
        if V is not None:
            # Fase A
            sig = V[:, 0]
            V1, V3, thd_val, freqs, mag = calculate_fft_thd(sig, fs)
            
            stats.append({
                "Cenário": item["label"],
                "V1 (V)": f"{V1:.2f}",
                "V3 (V)": f"{V3:.2f}",
                "V3/V1 (%)": f"{(V3/V1*100):.2f}",
                "THD (%)": f"{thd_val:.2f}"
            })
            
            # Plotar apenas até 500Hz para clareza
            mask_freq = freqs <= 500
            fig.add_trace(go.Scatter(x=freqs[mask_freq], y=mag[mask_freq], mode='lines', name=item["label"]))
            
    st.plotly_chart(fig, use_container_width=True)
    
    if calc_thd:
        st.write("### Tabela de Métricas Harmônicas")
        st.dataframe(pd.DataFrame(stats))


elif analise_tipo == "4.11 - Razão V3/V1 (%)":
    st.subheader("Figura 4.11 - Razão V3/V1 (%) por Cenário")
    # Similar ao 4.8 mas com V3/V1
    labels, vals = [], []
    for item in data_store:
        df = item["df"]
        fs = item["fs"]
        mask = (df.iloc[:,0] >= janela[0]) & (df.iloc[:,0] <= janela[1])
        V, _ = get_bus_data(df.loc[mask], barra_selecionada)
        if V is not None:
            V1, V3, _, _, _ = calculate_fft_thd(V[:, 0], fs)
            ratio = (V3/V1)*100 if V1 > 0 else 0
            labels.append(item["label"])
            vals.append(ratio)
            
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color='indianred'))
    fig.update_layout(title=f"Distorção de 3ª Harmônica - Barra {barra_selecionada}", yaxis_title="V3/V1 (%)")
    st.plotly_chart(fig, use_container_width=True)


elif analise_tipo == "4.12 - Corrente Máxima de Falta":
    st.subheader("Figura 4.12 - Corrente Máxima de Falta (Janela Selecionada)")
    # Calcula o MAX do RMS dentro da janela
    
    labels, imax_vals = [], []
    for item in data_store:
        df = item["df"]
        fs = item["fs"]
        mask = (df.iloc[:,0] >= janela[0]) & (df.iloc[:,0] <= janela[1])
        df_win = df.loc[mask]
        
        _, I = get_bus_data(df_win, barra_selecionada)
        if I is not None:
            # RMS da janela inteira ou pico do RMS deslizante na janela?
            # Geralmente é o pico do RMS durante a falta
            i_rms_abc = [np.max(sliding_rms(I[:, i], fs)) for i in range(3)]
            max_current = np.max(i_rms_abc)
            
            labels.append(item["label"])
            imax_vals.append(max_current)
            
    fig = go.Figure(go.Bar(x=labels, y=imax_vals))
    fig.update_layout(title=f"I_Falta_Max - Barra {barra_selecionada}", yaxis_title="Corrente (A)")
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# 5. EXPORTAÇÃO
# ==============================================================================
st.markdown("---")
if st.button("Exportar Dados da Visualização Atual (CSV)"):
    # Lógica simples para baixar o que foi calculado (pode ser expandido)
    st.info("Funcionalidade de exportação pronta para implementação baseada no DataFrame exibido acima.")
