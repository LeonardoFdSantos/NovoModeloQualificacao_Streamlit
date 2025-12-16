import io
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.io import loadmat
from scipy.signal import windows
from sklearn.tree import DecisionTreeClassifier

# =========================================================
# CONFIGURAÇÃO VISUAL (TEMA DARK NEON)
# =========================================================
st.set_page_config(
    page_title="PhD T2F Analysis - Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paleta de Cores de Alto Contraste
THEME = {
    "bg": "#0e1117", 
    "plot_bg": "#0e1117", 
    "grid": "rgba(255,255,255,0.1)",
    "text": "#e6edf3",
    "A": "#00d4ff", # Ciano Neon
    "B": "#ff4b4b", # Vermelho Neon
    "C": "#5aff5a", # Verde Neon
    "V1": "#29b5e8", 
    "V2": "#e8299c", # Magenta
    "V0": "#fcc203", # Amarelo Ouro
    "vec": "#ffffff"
}

ANGLES = np.deg2rad([0, 120, 240])
DATASET_FILE = "dataset_faltas_t2f_final.csv"

# =========================================================
# 1. TRATAMENTO DE DADOS (ROBUSTEZ)
# =========================================================
@st.cache_data(show_spinner=False)
def load_mat_file(file_bytes):
    try: return loadmat(io.BytesIO(file_bytes), squeeze_me=False, struct_as_record=False)
    except Exception as e: return str(e)

def safe_downsample(t, x, max_points=3000):
    """
    Reduz a densidade de pontos para visualização (Performance),
    mas mantém a forma da onda. Evita travar o navegador.
    """
    if t is None or len(t) == 0: return [], []
    if len(t) <= max_points: return t, x
    
    # Garante fator inteiro >= 1
    factor = int(len(t) / max_points)
    if factor < 1: factor = 1
    
    return t[::factor], x[::factor]

def extract_keys_safely(mat_data):
    if 'ts' not in mat_data: return []
    ts_struct = mat_data['ts']
    if isinstance(ts_struct, np.ndarray) and ts_struct.size == 1: ts_struct = ts_struct.item()
    
    raw_keys = []
    if hasattr(ts_struct, '_fieldnames'): raw_keys = ts_struct._fieldnames
    elif hasattr(ts_struct, 'dtype') and getattr(ts_struct.dtype, 'names', None): raw_keys = ts_struct.dtype.names
    elif isinstance(ts_struct, dict): raw_keys = ts_struct.keys()
    else: raw_keys = [k for k in dir(ts_struct) if not k.startswith('_')]
        
    return sorted([str(k).replace("ts_", "") for k in raw_keys])

def extract_signal(mat, point_name):
    if 'ts' not in mat: return None, None
    ts_root = mat['ts']
    if isinstance(ts_root, np.ndarray): 
        if ts_root.size == 1: ts_root = ts_root.item()
        else: ts_root = ts_root[0]
    
    key = f"ts_{point_name}"
    # Tenta acesso robusto
    entry = ts_root.get(key) if isinstance(ts_root, dict) else getattr(ts_root, key, None)
    
    if entry is None: return None, None
    if isinstance(entry, np.ndarray) and entry.size == 1: entry = entry.item()
    
    t = getattr(entry, "Time", getattr(entry, "time", None))
    x = getattr(entry, "Data", getattr(entry, "data", None))
    
    if t is None or x is None: return None, None

    t = np.asarray(t).squeeze()
    x = np.asarray(x).squeeze()
    
    # Corrige orientação (N, 3)
    if x.ndim == 2:
        if x.shape[0] == 3 and x.shape[1] > 3: x = x.T
        
    return t, x

# =========================================================
# 2. MACHINE LEARNING (CLASSIFICADOR T2F)
# =========================================================
def train_model():
    if not os.path.exists(DATASET_FILE): return None
    try:
        df = pd.read_csv(DATASET_FILE)
        if len(df) < 3: return None # Mínimo para começar
        X = df[['r0', 'r2', 'v1_mag', 'i1_mag']]
        y = df['label']
        clf = DecisionTreeClassifier(max_depth=5, random_state=42)
        clf.fit(X, y)
        return clf
    except: return None

def classify_fault_hybrid(V0, V1, V2, I0, I1, I2, thresh_0, thresh_2):
    i1_mag = abs(I1); denom = i1_mag if i1_mag > 1e-3 else 1e-3
    r0 = abs(I0) / denom; r2 = abs(I2) / denom
    
    # Features garantidas como float puro (evita erro de serialização)
    features = {
        'r0': float(r0), 'r2': float(r2), 
        'v1_mag': float(abs(V1)), 'i1_mag': float(i1_mag)
    }

    model = train_model()
    if model:
        try:
            pred = model.predict(pd.DataFrame([features]))[0]
            return pred, "#2196F3", f"🤖 IA T2F ({len(pd.read_csv(DATASET_FILE))} casos)", features
        except:
            pass # Fallback para manual se o modelo falhar

    # Heurística T2F (Baseada na sua Tese)
    if i1_mag < 0.1: return "Sem Carga", "gray", "Manual", features
    if r0 > thresh_0 and r2 > thresh_2:
        if 0.8 < (r0/r2) < 1.2: return "Bifásico Terra (AC/BC)", "#E91E63", "Manual", features
    if r2 > thresh_2 and r0 < thresh_0: return "Bifásico Aéreo (AB)", "#FF9800", "Manual", features
    if r0 < thresh_0 and r2 < thresh_2: return "Normal / Trifásico", "#4CAF50", "Manual", features
    return "Indeterminado", "gray", "Manual", features

def save_training_point(features, true_label):
    data = features.copy(); data['label'] = true_label
    df_new = pd.DataFrame([data])
    if os.path.exists(DATASET_FILE): df_new.to_csv(DATASET_FILE, mode='a', header=False, index=False)
    else: df_new.to_csv(DATASET_FILE, index=False)

# =========================================================
# 3. MATEMÁTICA (CACHEADA)
# =========================================================
def clarke_transform(a, b, c, mode="power"):
    k = (2/3) if mode == "amp" else np.sqrt(2/3)
    alpha = k * (a - 0.5*b - 0.5*c)
    beta  = k * ((np.sqrt(3)/2)*b - (np.sqrt(3)/2)*c)
    return alpha, beta

def phasor_from_signal(x, t, f0=60.0):
    x = x - np.mean(x); N = len(x)
    if N < 4: return 0j
    window = windows.hann(N); xw = x * window
    X = np.sum(xw * np.exp(-1j * 2*np.pi*f0*t))
    return (2.0 * X / np.sum(window)) / np.sqrt(2)

def sym_comp_phasors(Va, Vb, Vc):
    a = np.exp(1j * 2*np.pi/3)
    T = (1/3) * np.array([[1, 1, 1], [1, a**2, a], [1, a, a**2]], dtype=complex)
    return T @ np.array([Va, Vb, Vc])

def calculate_impedance(Va, Vb, Vc, Ia, Ib, Ic):
    # Cálculo seguro contra divisão por zero
    with np.errstate(divide='ignore', invalid='ignore'):
        Za = np.where(np.abs(Ia)>1e-2, Va/Ia, 0j)
        Zb = np.where(np.abs(Ib)>1e-2, Vb/Ib, 0j)
        Zc = np.where(np.abs(Ic)>1e-2, Vc/Ic, 0j)
    return Za, Zb, Zc

def synth_phasor_time(Vrms, t, f0):
    # Retorna APENAS a parte real para plotagem (evita erros de complexo no plotly)
    return np.sqrt(2) * np.real(Vrms * np.exp(1j * 2*np.pi*f0*t))

@st.cache_data(show_spinner=False)
def process_full_analysis(t, va, vb, vc, ia, ib, ic, f0, clarke_mode, t0, t2):
    # 1. Clarke
    alpha_v, beta_v = clarke_transform(va, vb, vc, mode=clarke_mode)
    alpha_i, beta_i = clarke_transform(ia, ib, ic, mode=clarke_mode)
    
    # 2. Fasores
    Va_ph = phasor_from_signal(va, t, f0); Vb_ph = phasor_from_signal(vb, t, f0); Vc_ph = phasor_from_signal(vc, t, f0)
    Ia_ph = phasor_from_signal(ia, t, f0); Ib_ph = phasor_from_signal(ib, t, f0); Ic_ph = phasor_from_signal(ic, t, f0)
    
    V0, V1, V2 = sym_comp_phasors(Va_ph, Vb_ph, Vc_ph)
    I0, I1, I2 = sym_comp_phasors(Ia_ph, Ib_ph, Ic_ph)
    
    # 3. Classificação e Impedância
    desc, color, method, features = classify_fault_hybrid(V0, V1, V2, I0, I1, I2, t0, t2)
    Za, Zb, Zc = calculate_impedance(va, vb, vc, ia, ib, ic)
    
    # 4. Reconstrução Temporal (Sequências)
    a_op = np.exp(1j * 2*np.pi/3)
    Ti = np.array([[1, 1, 1], [1, a_op, a_op**2], [1, a_op**2, a_op]], dtype=complex)
    
    def get_seq_data(C0, C1, C2):
        v_a0, _, _ = Ti @ np.array([C0, 0, 0]); s0 = synth_phasor_time(v_a0, t, f0)
        v_a1, _, _ = Ti @ np.array([0, C1, 0]); s1 = synth_phasor_time(v_a1, t, f0)
        v_a2, _, _ = Ti @ np.array([0, 0, C2]); s2 = synth_phasor_time(v_a2, t, f0)
        return s0, s1, s2

    s0v, s1v, s2v = get_seq_data(V0, V1, V2)
    s0i, s1i, s2i = get_seq_data(I0, I1, I2)

    return {
        "t": t, "v": (va, vb, vc), "i": (ia, ib, ic),
        "clarke_v": (alpha_v, beta_v), "clarke_i": (alpha_i, beta_i),
        "phasors_v": (V0, V1, V2), "phasors_i": (I0, I1, I2),
        "seqs_v": (s0v, s1v, s2v), "seqs_i": (s0i, s1i, s2i),
        "fault_info": (desc, color, method),
        "features": features, 
        "Z_traj": (Za, Zb, Zc)
    }

# =========================================================
# 4. PLOTAGEM ESTÁVEL (SVG + DECIMAÇÃO)
# =========================================================
def plot_animation_robust(data, mode):
    t = data["t"]
    
    # Seleção de Dados
    if mode == "Tensão":
        a, b, c = data["v"]; av, bv = data["clarke_v"]; s0, s1, s2 = data["seqs_v"]
        phasors = data["phasors_v"]
    else:
        a, b, c = data["i"]; av, bv = data["clarke_i"]; s0, s1, s2 = data["seqs_i"]
        phasors = data["phasors_i"]

    # Decimação (Reduzir pontos para não travar o Plotly)
    t_vis, a_vis = safe_downsample(t, a); _, b_vis = safe_downsample(t, b); _, c_vis = safe_downsample(t, c)
    _, av_vis = safe_downsample(t, av); _, bv_vis = safe_downsample(t, bv)
    _, s0_vis = safe_downsample(t, s0); _, s1_vis = safe_downsample(t, s1); _, s2_vis = safe_downsample(t, s2)

    # Layout Matriz 2x3 (Tempo, Fasor, Clarke, Seqs, 3D)
    fig = make_subplots(rows=2, cols=3, 
        specs=[[{"colspan": 2}, None, {"type": "polar"}], 
               [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=(f"Formas de Onda {mode}", f"Fasores {mode} (Fund.)", 
                        "Plano Clarke (αβ)", "Sequências (Tempo)", "Análise Espacial (Alpha-Beta-Time)"),
        horizontal_spacing=0.08, vertical_spacing=0.15)
    
    # 1. TEMPO (SVG Otimizado)
    fig.add_trace(go.Scatter(x=t_vis, y=a_vis, name="A", line=dict(color=THEME["A"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_vis, y=b_vis, name="B", line=dict(color=THEME["B"], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_vis, y=c_vis, name="C", line=dict(color=THEME["C"], width=1.5)), row=1, col=1)

    # 2. FASORES (Polar)
    p0, p1, p2 = phasors
    fig.add_trace(go.Scatterpolar(r=[0, abs(p1)], theta=[0, np.angle(p1, deg=True)], name="Pos", line_color=THEME["V1"]), row=1, col=3)
    fig.add_trace(go.Scatterpolar(r=[0, abs(p2)], theta=[0, np.angle(p2, deg=True)], name="Neg", line_color=THEME["V2"]), row=1, col=3)
    fig.add_trace(go.Scatterpolar(r=[0, abs(p0)], theta=[0, np.angle(p0, deg=True)], name="Zero", line_color=THEME["V0"]), row=1, col=3)

    # 3. CLARKE (XY)
    fig.add_trace(go.Scatter(x=av_vis, y=bv_vis, mode='lines', name="αβ", line=dict(color="white", width=1)), row=2, col=1)

    # 4. SEQUENCIAS (Tempo)
    fig.add_trace(go.Scatter(x=t_vis, y=s1_vis, name="V1", line=dict(color=THEME["V1"], width=1)), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_vis, y=s2_vis, name="V2", line=dict(color=THEME["V2"], width=1)), row=2, col=2)
    fig.add_trace(go.Scatter(x=t_vis, y=s0_vis, name="V0", line=dict(color=THEME["V0"], width=1)), row=2, col=2)

    # 5. 3D SIMULADO (Scatter 2D colorido pelo tempo)
    # Isso é mais leve que um gráfico 3D real e mostra a mesma informação de evolução
    fig.add_trace(go.Scatter(
        x=av_vis, y=bv_vis, 
        mode='markers', 
        marker=dict(size=3, color=np.arange(len(av_vis)), colorscale='Viridis', showscale=False), 
        name="Evolução"
    ), row=2, col=3)

    # Layout Global Dark
    fig.update_layout(
        height=700, 
        template="plotly_dark", 
        paper_bgcolor=THEME["bg"], 
        plot_bgcolor=THEME["plot_bg"],
        font=dict(color=THEME["text"]),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=-0.1)
    )
    
    # Eixos Polares
    fig.update_layout(polar=dict(bgcolor=THEME["plot_bg"], radialaxis=dict(showticklabels=False, gridcolor=THEME["grid"])))
    
    return fig

def plot_3d_trajectory_real(alpha, beta, t):
    """Gráfico 3D Real para análise profunda."""
    # Decimação agressiva para 3D (máx 2000 pontos) para não travar rotação
    t_v, _ = safe_downsample(t, t, 2000)
    a_v, _ = safe_downsample(t, alpha, 2000)
    b_v, _ = safe_downsample(t, beta, 2000)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=a_v, y=b_v, z=t_v,
        mode='lines',
        line=dict(color=t_v, colorscale='Turbo', width=5),
        name='Trajetória'
    )])
    
    fig.update_layout(
        title="Espiral de Clarke no Tempo",
        scene=dict(
            xaxis_title='Alpha', yaxis_title='Beta', zaxis_title='Tempo (s)',
            xaxis=dict(backgroundcolor=THEME["plot_bg"], gridcolor=THEME["grid"]),
            yaxis=dict(backgroundcolor=THEME["plot_bg"], gridcolor=THEME["grid"]),
            zaxis=dict(backgroundcolor=THEME["plot_bg"], gridcolor=THEME["grid"]),
        ),
        height=700, 
        template="plotly_dark",
        paper_bgcolor=THEME["bg"]
    )
    return fig

# =========================================================
# 5. APP PRINCIPAL
# =========================================================
st.title("⚡ Análise T2F Avançada (PhD Tool)")

with st.sidebar:
    st.markdown("### 1. Dados de Entrada")
    uploaded_files = st.file_uploader("Arquivos .mat", type=["mat"], accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("Aguardando arquivos...")
        st.stop()
    
    # Carrega todos os arquivos
    mats = {}
    for uf in uploaded_files:
        res = load_mat_file(uf.getvalue())
        if not isinstance(res, str): mats[uf.name] = res
    
    if not mats: st.error("Nenhum arquivo válido lido."); st.stop()
    
    sel = st.selectbox("Arquivo Ativo", sorted(mats.keys()))
    mat = mats[sel]
    keys = extract_keys_safely(mat)
    
    if not keys: st.error("Arquivo sem estrutura 'ts' compatível."); st.stop()
    
    vp = st.selectbox("Canal Tensão", [k for k in keys if k.startswith('V')] or keys, index=0)
    ip = st.selectbox("Canal Corrente", [k for k in keys if k.startswith('I')] or keys, index=0)
    
    st.divider()
    view = st.radio("Modo de Visualização", ["Dashboard Geral", "Análise 3D Espacial"])

# --- PROCESSAMENTO ---
t_v, vr = extract_signal(mat, vp)
t_i, ir = extract_signal(mat, ip)

if t_v is None or t_i is None or len(t_v) == 0: 
    st.error("Erro na leitura dos sinais. Verifique o nome das variáveis."); st.stop()

# Sincronização
nm = min(len(t_v), len(t_i))
t = t_v[:nm]; va, vb, vc = vr[:nm].T; ia, ib, ic = ir[:nm].T

# Remove DC
va -= np.mean(va); vb -= np.mean(vb); vc -= np.mean(vc)
ia -= np.mean(ia); ib -= np.mean(ib); ic -= np.mean(ic)

# Executa matemática
data = process_full_analysis(t, va, vb, vc, ia, ib, ic, 60.0, "power", 0.15, 0.15)

# --- CABEÇALHO INTELIGENTE ---
desc, col, met, feats = data["fault_info"][0], data["fault_info"][1], data["fault_info"][2], data["features"]

c1, c2, c3, c4 = st.columns([3, 1, 1, 2])
c1.markdown(f"<div style='padding:15px; border-radius:10px; background:{col}22; border-left:5px solid {col}'>"
            f"<h3 style='margin:0; color:{col}'>{desc}</h3><small>{met}</small></div>", unsafe_allow_html=True)
c2.metric("I2/I1 (Neg)", f"{feats['r2']:.3f}")
c3.metric("I0/I1 (Zero)", f"{feats['r0']:.3f}")

with c4:
    lbl_opts = ["Normal", "Trifásico C1", "Trifásico C2", "Bifásico AB", "Bifásico AC", "Bifásico BC"]
    lbl = st.selectbox("Treinar Classificador:", lbl_opts, label_visibility="collapsed")
    if st.button("Salvar Treino"): 
        save_training_point(feats, lbl)
        st.cache_data.clear()
        st.rerun()

# --- VISUALIZAÇÃO ---
if view == "Dashboard Geral":
    st.subheader("Análise de Tensão")
    # KEY ÚNICA PARA EVITAR O ERRO 'Duplicate ID'
    st.plotly_chart(plot_animation_robust(data, "Tensão"), use_container_width=True, key="chart_voltage_main")
    
    st.subheader("Análise de Corrente")
    st.plotly_chart(plot_animation_robust(data, "Corrente"), use_container_width=True, key="chart_current_main")

elif view == "Análise 3D Espacial":
    st.info("Visualização 3D da trajetória do vetor de Clarke (Alpha-Beta) ao longo do tempo (Eixo Z).")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Trajetória Tensão**")
        st.plotly_chart(plot_3d_trajectory_real(data["clarke_v"][0], data["clarke_v"][1], t), use_container_width=True, key="3d_v")
    with c2:
        st.markdown("**Trajetória Corrente**")
        st.plotly_chart(plot_3d_trajectory_real(data["clarke_i"][0], data["clarke_i"][1], t), use_container_width=True, key="3d_i")