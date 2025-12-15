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
# CONFIGURAÇÃO DA PÁGINA
# =========================================================
st.set_page_config(
    page_title="PhD T2F Protection & Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes Visuais
ANGLES = np.deg2rad([0, 120, 240])  # A,B,C
PAPER_BG = "#0f1117"
PLOT_BG  = "#0f1117"
GRID_CLR = "rgba(255,255,255,0.08)"
FONT_CLR = "#e6edf3"
DATASET_FILE = "dataset_faltas_t2f.csv" # Arquivo onde o aprendizado será salvo

# =========================================================
# 1. FUNÇÕES DE SUPORTE (DATA LOADING)
# =========================================================
@st.cache_data(show_spinner=False)
def load_mat_file(file_bytes):
    try:
        return loadmat(io.BytesIO(file_bytes), squeeze_me=False, struct_as_record=False)
    except Exception as e:
        return str(e)

def extract_keys_safely(mat_data):
    if 'ts' not in mat_data: return []
    ts_struct = mat_data['ts']
    if isinstance(ts_struct, np.ndarray) and ts_struct.size == 1: 
        ts_struct = ts_struct.item()
    
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
    # Tenta acesso via dict ou atributo
    entry = ts_root.get(key) if isinstance(ts_root, dict) else getattr(ts_root, key, None)
    
    if entry is None: return None, None
    if isinstance(entry, np.ndarray) and entry.size == 1: entry = entry.item()
    
    t = getattr(entry, "Time", getattr(entry, "time", None))
    x = getattr(entry, "Data", getattr(entry, "data", None))
    
    if t is None or x is None: return None, None

    t = np.asarray(t).squeeze()
    x = np.asarray(x).squeeze()
    
    # Garante formato (N, 3)
    if x.ndim == 2:
        if x.shape[0] == 3 and x.shape[1] > 3: x = x.T
        
    return t, x

# =========================================================
# 2. INTEGRAÇÃO MACHINE LEARNING (TREINAMENTO T2F)
# =========================================================

def train_model():
    """Treina uma Árvore de Decisão se houver dados suficientes."""
    if not os.path.exists(DATASET_FILE):
        return None
    
    try:
        df = pd.read_csv(DATASET_FILE)
        # Precisamos de pelo menos 3 classes diferentes ou 5 exemplos para começar a arriscar
        if len(df) < 3: 
            return None
            
        # Features usadas para decisão
        X = df[['r0', 'r2', 'v1_mag', 'i1_mag']]
        y = df['label']
        
        clf = DecisionTreeClassifier(max_depth=6, random_state=42)
        clf.fit(X, y)
        return clf
    except:
        return None

def classify_fault_hybrid(V0, V1, V2, I0, I1, I2, thresh_0, thresh_2):
    """Classifica usando ML (se disponível) ou Heurística (Sliders)."""
    i1_mag = abs(I1)
    denom = i1_mag if i1_mag > 1e-3 else 1e-3
    
    r0 = abs(I0) / denom # Razão Zero/Positiva
    r2 = abs(I2) / denom # Razão Negativa/Positiva
    
    # Dados para o modelo
    features = {
        'r0': float(r0),
        'r2': float(r2),
        'v1_mag': float(abs(V1)),
        'i1_mag': float(i1_mag)
    }

    model = train_model()
    
    if model:
        # --- MODO MACHINE LEARNING ---
        # O modelo decide com base no que você ensinou
        pred = model.predict(pd.DataFrame([features]))[0]
        return pred, "#2196F3", f"🤖 IA T2F (Treinada com {len(pd.read_csv(DATASET_FILE))} casos)", features
    else:
        # --- MODO MANUAL (HEURÍSTICA) ---
        # Usa os sliders da sidebar
        if i1_mag < 0.1: return "Sem Carga / Desligado", "gray", "Manual", features
        
        is_zero = r0 > thresh_0
        is_neg  = r2 > thresh_2

        if not is_zero and not is_neg:
            return "Normal / Carga Equilibrada", "#4CAF50", "Manual", features
        
        # Lógica genérica que será substituída pelo seu treino
        if not is_zero and is_neg:
            return "Bifásico Aéreo (AB) - Provável", "#FF9800", "Manual", features
        
        if is_zero and is_neg:
            # Tenta distinguir terra vs bifásico terra
            if 0.8 < (r0/r2) < 1.2:
                return "Bifásico Terra (AC ou BC) - Provável", "#E91E63", "Manual", features
            else:
                return "Falta Complexa (Treine o modelo!)", "#9C27B0", "Manual", features
                
        if is_zero and not is_neg:
            return "Falta Envolvendo Terra (I0 puro)", "#673AB7", "Manual", features

        return "Indeterminado", "gray", "Manual", features

def save_training_point(features, true_label):
    """Salva a correção no CSV."""
    data = features.copy()
    data['label'] = true_label
    df_new = pd.DataFrame([data])
    
    if os.path.exists(DATASET_FILE):
        df_new.to_csv(DATASET_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(DATASET_FILE, index=False)

# =========================================================
# 3. CÁLCULOS MATEMÁTICOS (CACHEADOS)
# =========================================================

def clarke_transform(a, b, c, mode="power"):
    k = (2/3) if mode == "amp" else np.sqrt(2/3)
    alpha = k * (a - 0.5*b - 0.5*c)
    beta  = k * ((np.sqrt(3)/2)*b - (np.sqrt(3)/2)*c)
    return alpha, beta

def phasor_from_signal(x, t, f0=60.0):
    x = x - np.mean(x)
    N = len(x)
    if N < 4: return 0j
    window = windows.hann(N)
    xw = x * window
    # Projeção DFT na fundamental
    exp_term = np.exp(-1j * 2*np.pi*f0*t)
    X = np.sum(xw * exp_term)
    W = np.sum(window)
    return (2.0 * X / W) / np.sqrt(2) if W != 0 else 0j

def sym_comp_phasors(Va, Vb, Vc):
    a = np.exp(1j * 2*np.pi/3)
    T = (1/3) * np.array([[1, 1, 1], [1, a**2, a], [1, a, a**2]], dtype=complex)
    return T @ np.array([Va, Vb, Vc])

def calculate_impedance(Va, Vb, Vc, Ia, Ib, Ic):
    mask = np.abs(Ia) > 1e-2
    Za = np.zeros_like(Va, dtype=complex); Za[mask] = Va[mask] / Ia[mask]
    mask = np.abs(Ib) > 1e-2
    Zb = np.zeros_like(Vb, dtype=complex); Zb[mask] = Vb[mask] / Ib[mask]
    mask = np.abs(Ic) > 1e-2
    Zc = np.zeros_like(Vc, dtype=complex); Zc[mask] = Vc[mask] / Ic[mask]
    return Za, Zb, Zc

def synth_sequence(Vrms, t, f0):
    # Reconstrói onda no tempo baseada no fasor
    return np.sqrt(2) * np.real(Vrms * np.exp(1j * 2*np.pi*f0*t))

@st.cache_data(show_spinner=False)
def process_full_analysis(t, va, vb, vc, ia, ib, ic, f0, clarke_mode, t0, t2):
    # 1. Clarke (V e I)
    alpha_v, beta_v = clarke_transform(va, vb, vc, mode=clarke_mode)
    alpha_i, beta_i = clarke_transform(ia, ib, ic, mode=clarke_mode)
    
    # 2. Fasores e Sequências (Fundamental)
    Va_ph = phasor_from_signal(va, t, f0); Vb_ph = phasor_from_signal(vb, t, f0); Vc_ph = phasor_from_signal(vc, t, f0)
    Ia_ph = phasor_from_signal(ia, t, f0); Ib_ph = phasor_from_signal(ib, t, f0); Ic_ph = phasor_from_signal(ic, t, f0)
    
    V0, V1, V2 = sym_comp_phasors(Va_ph, Vb_ph, Vc_ph)
    I0, I1, I2 = sym_comp_phasors(Ia_ph, Ib_ph, Ic_ph)
    
    # 3. Classificação Híbrida (ML ou Manual)
    desc, color, method, features = classify_fault_hybrid(V0, V1, V2, I0, I1, I2, t0, t2)
    
    # 4. Impedância
    Za, Zb, Zc = calculate_impedance(va, vb, vc, ia, ib, ic)
    
    # 5. Reconstrução Temporal das Sequências (para animação V e I)
    a_op = np.exp(1j * 2*np.pi/3)
    Ti = np.array([[1, 1, 1], [1, a_op, a_op**2], [1, a_op**2, a_op]], dtype=complex)
    
    def get_seq_curves(Comp0, Comp1, Comp2):
        # Gera curvas para cada componente isolada (apenas para visualização)
        v_a0, _, _ = Ti @ np.array([Comp0, 0, 0])
        v_a1, _, _ = Ti @ np.array([0, Comp1, 0])
        v_a2, _, _ = Ti @ np.array([0, 0, Comp2])
        return {
            "s0": synth_sequence(v_a0, t, f0),
            "s1": synth_sequence(v_a1, t, f0),
            "s2": synth_sequence(v_a2, t, f0)
        }

    return {
        "alpha_v": alpha_v, "beta_v": beta_v, "alpha_i": alpha_i, "beta_i": beta_i,
        "V_phasors": (V0, V1, V2), "I_phasors": (I0, I1, I2),
        "fault_info": (desc, color, method),
        "features": features,
        "Z_traj": (Za, Zb, Zc),
        "seq_curves_v": get_seq_curves(V0, V1, V2),
        "seq_curves_i": get_seq_curves(I0, I1, I2)
    }

# =========================================================
# 4. INTERFACE DE USUÁRIO (UI)
# =========================================================
st.markdown("## ⚡ Análise de Faltas T2F - PhD Tool")

with st.sidebar:
    st.header("1. Arquivos")
    uploaded_files = st.file_uploader("Arraste arquivos .mat", type=["mat"], accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("Aguardando arquivos...")
        st.stop()

    loaded_mats = {}
    for uf in uploaded_files:
        data = load_mat_file(uf.getvalue())
        if not isinstance(data, str): loaded_mats[uf.name] = data
    
    if not loaded_mats: st.error("Nenhum arquivo válido."); st.stop()
    
    selected_filename = st.selectbox("Arquivo Ativo", sorted(loaded_mats.keys()))
    mat_data = loaded_mats[selected_filename]
    all_keys = extract_keys_safely(mat_data)
    
    st.divider()
    st.header("2. Sinais")
    v_keys = [k for k in all_keys if k.startswith('V')]; i_keys = [k for k in all_keys if k.startswith('I')]
    v_point = st.selectbox("Canal Tensão (V)", v_keys if v_keys else all_keys, index=0)
    i_point = st.selectbox("Canal Corrente (I)", i_keys if i_keys else all_keys, index=0)
    
    st.divider()
    st.header("3. Ajustes")
    remove_mean = st.checkbox("Remover DC", value=True)
    frame_step = st.slider("Velocidade Animação", 1, 60, 10)
    
    # Sliders só aparecem se não tiver modelo treinado (ou para debug)
    with st.expander("Calibração Manual (Se não houver ML)"):
        thresh_0 = st.slider("Limiar Seq Zero (I0/I1)", 0.01, 1.0, 0.15, 0.01)
        thresh_2 = st.slider("Limiar Seq Neg (I2/I1)", 0.01, 1.0, 0.15, 0.01)

# --- PROCESSAMENTO PRINCIPAL ---
t_v, v_raw = extract_signal(mat_data, v_point)
t_i, i_raw = extract_signal(mat_data, i_point)

if t_v is None or t_i is None: st.error("Erro nos dados."); st.stop()

n_min = min(len(t_v), len(t_i))
t = t_v[:n_min]; va, vb, vc = v_raw[:n_min].T; ia, ib, ic = i_raw[:n_min].T

if remove_mean:
    va -= np.mean(va); vb -= np.mean(vb); vc -= np.mean(vc)
    ia -= np.mean(ia); ib -= np.mean(ib); ic -= np.mean(ic)

# Roda análise completa
data = process_full_analysis(t, va, vb, vc, ia, ib, ic, 60.0, "power", thresh_0, thresh_2)

# =========================================================
# ÁREA DE DIAGNÓSTICO E TREINAMENTO
# =========================================================
features = data["features"]
desc, color, method = data["fault_info"]

c1, c2, c3, c4 = st.columns([2, 1, 1, 2])

with c1:
    st.markdown(f"""
    <div style="padding:15px; border-radius:8px; background:{color}22; border-left: 5px solid {color}">
        <h3 style="margin:0; color:{color}">{desc}</h3>
        <small>Método: <b>{method}</b> | Arquivo: {selected_filename}</small>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.metric("Razão I2/I1 (Neg)", f"{features['r2']:.3f}")
with c3:
    st.metric("Razão I0/I1 (Zero)", f"{features['r0']:.3f}")

with c4:
    st.markdown("#### 🎯 Ensinar o Sistema (T2F)")
    # TIPOS DE FALTA DA SUA TESE
    tipos_t2f = [
        "Normal / Carga Equilibrada",
        "Trifásico ABC - Caso 1 (Solo)",
        "Trifásico ABC - Caso 2 (Aéreo)",
        "Bifásico Aéreo (AB)",
        "Bifásico Terra (AC)",
        "Bifásico Terra (BC)"
    ]
    correct_label = st.selectbox("Classificação Real:", types_t2f := tipos_t2f, label_visibility="collapsed")
    
    if st.button("💾 Salvar Treinamento"):
        save_training_point(features, correct_label)
        st.success("Aprendido! O sistema agora reconhecerá este padrão.")
        st.cache_data.clear() # Limpa cache para o modelo atualizar

# =========================================================
# ABAS VISUAIS
# =========================================================
tab1, tab2 = st.tabs(["📺 Animação V & I (Tempo Real)", "🛡️ Proteção (Impedância & Fasores)"])

with tab1:
    # Preparação dos dados para plotagem
    alpha_v, beta_v = data["alpha_v"], data["beta_v"]
    seqs_v = data["seq_curves_v"]
    alpha_i, beta_i = data["alpha_i"], data["beta_i"]
    seqs_i = data["seq_curves_i"]
    
    indices = list(range(0, len(t), frame_step))
    if indices[-1] != len(t)-1: indices.append(len(t)-1)
    
    # Layout 3x2 (V esquerda, I direita)
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=("Tensão ABC", "Corrente ABC", "Tensão Sequências", "Corrente Sequências", "Vetor Espacial V", "Vetor Espacial I"),
        vertical_spacing=0.08, horizontal_spacing=0.05
    )
    
    # Cores Consistentes
    c_a, c_b, c_c = "#FF5252", "#4CAF50", "#448AFF"
    c_s1, c_s2, c_s0 = "cyan", "orange", "yellow"

    # --- TRACES DE FUNDO (ESTÁTICOS) ---
    # Tensão
    fig.add_trace(go.Scatter(x=t, y=va, line=dict(color=c_a, width=1), name="Va"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=vb, line=dict(color=c_b, width=1), name="Vb"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=vc, line=dict(color=c_c, width=1), name="Vc"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=seqs_v['s1'], line=dict(color=c_s1, width=1), name="V1"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=seqs_v['s2'], line=dict(color=c_s2, width=1, dash='dash'), name="V2"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=seqs_v['s0'], line=dict(color=c_s0, width=1, dash='dot'), name="V0"), row=2, col=1)
    fig.add_trace(go.Scatter(x=alpha_v, y=beta_v, line=dict(color="rgba(255,255,255,0.2)", width=1), showlegend=False), row=3, col=1)

    # Corrente
    fig.add_trace(go.Scatter(x=t, y=ia, line=dict(color=c_a, width=1), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=ib, line=dict(color=c_b, width=1), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=ic, line=dict(color=c_c, width=1), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=t, y=seqs_i['s1'], line=dict(color=c_s1, width=1), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=seqs_i['s2'], line=dict(color=c_s2, width=1, dash='dash'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=t, y=seqs_i['s0'], line=dict(color=c_s0, width=1, dash='dot'), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=alpha_i, y=beta_i, line=dict(color="rgba(255,255,255,0.2)", width=1), showlegend=False), row=3, col=2)

    # --- TRACES DINÂMICOS (BOLINHAS E VETORES) ---
    # Indices: 14 traces estáticos acima (0 a 13). Dinâmicos começam no 14.
    # Adicionamos placeholders no frame 0
    # V: 3 dots ABC, 3 dots Seq, 1 vec AlphaBeta = 7 traces
    # I: 3 dots ABC, 3 dots Seq, 1 vec AlphaBeta = 7 traces
    
    # V Init
    fig.add_trace(go.Scatter(x=[t[0]], y=[va[0]], mode="markers", marker=dict(color="white", size=6), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[0]], y=[vb[0]], mode="markers", marker=dict(color="white", size=6), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[0]], y=[vc[0]], mode="markers", marker=dict(color="white", size=6), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[0]], y=[seqs_v['s1'][0]], mode="markers", marker=dict(color=c_s1, size=5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[t[0]], y=[seqs_v['s2'][0]], mode="markers", marker=dict(color=c_s2, size=5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[t[0]], y=[seqs_v['s0'][0]], mode="markers", marker=dict(color=c_s0, size=5), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0, alpha_v[0]], y=[0, beta_v[0]], mode="lines+markers", line=dict(color="white", width=3), showlegend=False), row=3, col=1)

    # I Init
    fig.add_trace(go.Scatter(x=[t[0]], y=[ia[0]], mode="markers", marker=dict(color="white", size=6), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[t[0]], y=[ib[0]], mode="markers", marker=dict(color="white", size=6), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[t[0]], y=[ic[0]], mode="markers", marker=dict(color="white", size=6), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=[t[0]], y=[seqs_i['s1'][0]], mode="markers", marker=dict(color=c_s1, size=5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[t[0]], y=[seqs_i['s2'][0]], mode="markers", marker=dict(color=c_s2, size=5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[t[0]], y=[seqs_i['s0'][0]], mode="markers", marker=dict(color=c_s0, size=5), showlegend=False), row=2, col=2)
    fig.add_trace(go.Scatter(x=[0, alpha_i[0]], y=[0, beta_i[0]], mode="lines+markers", line=dict(color="white", width=3), showlegend=False), row=3, col=2)

    frames = []
    for k in indices:
        frames.append(go.Frame(data=[
            # V Updates
            go.Scatter(x=[t[k]], y=[va[k]]), go.Scatter(x=[t[k]], y=[vb[k]]), go.Scatter(x=[t[k]], y=[vc[k]]),
            go.Scatter(x=[t[k]], y=[seqs_v['s1'][k]]), go.Scatter(x=[t[k]], y=[seqs_v['s2'][k]]), go.Scatter(x=[t[k]], y=[seqs_v['s0'][k]]),
            go.Scatter(x=[0, alpha_v[k]], y=[0, beta_v[k]]),
            # I Updates
            go.Scatter(x=[t[k]], y=[ia[k]]), go.Scatter(x=[t[k]], y=[ib[k]]), go.Scatter(x=[t[k]], y=[ic[k]]),
            go.Scatter(x=[t[k]], y=[seqs_i['s1'][k]]), go.Scatter(x=[t[k]], y=[seqs_i['s2'][k]]), go.Scatter(x=[t[k]], y=[seqs_i['s0'][k]]),
            go.Scatter(x=[0, alpha_i[k]], y=[0, beta_i[k]])
        ], name=str(k), traces=list(range(14, 28)))) # Atualiza os traces 14 até 27

    fig.frames = frames

    # Configuração de Layout
    fig.update_layout(
        height=1100, # AUMENTADO PARA TENSÃO FICAR VISÍVEL
        paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_CLR),
        legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"),
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.03, bgcolor="#1f2937", font=dict(color="white"),
            buttons=[dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 10}, "fromcurrent": True}]),
                     dict(label="⏸ Pause", method="animate", args=[[None], {"mode": "immediate"}])])],
        sliders=[dict(steps=[dict(method="animate", args=[[str(k)], {"mode":"immediate"}], label=f"{t[k]:.2f}") for k in indices[::max(1, len(indices)//50)]], 
                      currentvalue=dict(visible=True, prefix="Tempo: "), len=0.9, x=0.05, y=-0.02)]
    )
    
    # Fixar Eixos para não pular
    max_v = max(abs(va).max(), abs(vb).max(), abs(vc).max()) * 1.2
    max_i = max(abs(ia).max(), abs(ib).max(), abs(ic).max()) * 1.2
    
    # Eixos Tensão (Col 1)
    for r in [1, 2]: fig.update_yaxes(range=[-max_v, max_v], row=r, col=1)
    fig.update_xaxes(range=[-max_v, max_v], row=3, col=1); fig.update_yaxes(range=[-max_v, max_v], row=3, col=1, scaleanchor="x5", scaleratio=1)
    
    # Eixos Corrente (Col 2)
    for r in [1, 2]: fig.update_yaxes(range=[-max_i, max_i], row=r, col=2)
    fig.update_xaxes(range=[-max_i, max_i], row=3, col=2); fig.update_yaxes(range=[-max_i, max_i], row=3, col=2, scaleanchor="x6", scaleratio=1)

    st.plotly_chart(fig, use_container_width=True)

with tab2:
    cp1, cp2 = st.columns(2)
    with cp1:
        V0, V1, V2 = data["V_phasors"]
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatterpolar(r=[0, abs(V1)], theta=[0, np.angle(V1, deg=True)], name='V1 (Pos)', line_color='cyan'))
        fig_v.add_trace(go.Scatterpolar(r=[0, abs(V2)], theta=[0, np.angle(V2, deg=True)], name='V2 (Neg)', line_color='orange'))
        fig_v.add_trace(go.Scatterpolar(r=[0, abs(V0)], theta=[0, np.angle(V0, deg=True)], name='V0 (Zero)', line_color='yellow'))
        fig_v.update_layout(title="Fasores Tensão", polar=dict(radialaxis=dict(visible=True), bgcolor=PLOT_BG), paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR))
        st.plotly_chart(fig_v, use_container_width=True)
    
    with cp2:
        I0, I1, I2 = data["I_phasors"]
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatterpolar(r=[0, abs(I1)], theta=[0, np.angle(I1, deg=True)], name='I1 (Pos)', line_color='cyan'))
        fig_i.add_trace(go.Scatterpolar(r=[0, abs(I2)], theta=[0, np.angle(I2, deg=True)], name='I2 (Neg)', line_color='orange'))
        fig_i.add_trace(go.Scatterpolar(r=[0, abs(I0)], theta=[0, np.angle(I0, deg=True)], name='I0 (Zero)', line_color='yellow'))
        fig_i.update_layout(title="Fasores Corrente", polar=dict(radialaxis=dict(visible=True), bgcolor=PLOT_BG), paper_bgcolor=PAPER_BG, font=dict(color=FONT_CLR))
        st.plotly_chart(fig_i, use_container_width=True)
    
    st.divider()
    Za, Zb, Zc = data["Z_traj"]
    fig_rx = go.Figure()
    lim = 200; fz = lambda Z: Z[np.abs(Z)<lim]
    
    fig_rx.add_trace(go.Scatter(x=np.real(fz(Za)), y=np.imag(fz(Za)), mode='lines', name='Za', line=dict(color="#FF5252")))
    fig_rx.add_trace(go.Scatter(x=np.real(fz(Zb)), y=np.imag(fz(Zb)), mode='lines', name='Zb', line=dict(color="#4CAF50")))
    fig_rx.add_trace(go.Scatter(x=np.real(fz(Zc)), y=np.imag(fz(Zc)), mode='lines', name='Zc', line=dict(color="#448AFF")))
    
    fig_rx.update_layout(title="Plano de Impedância (R-X)", paper_bgcolor=PAPER_BG, plot_bgcolor=PLOT_BG, font=dict(color=FONT_CLR),
                         xaxis=dict(gridcolor=GRID_CLR, zeroline=True), yaxis=dict(gridcolor=GRID_CLR, zeroline=True, scaleanchor="x"), height=600)
    st.plotly_chart(fig_rx, use_container_width=True)