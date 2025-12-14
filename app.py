import io
import time
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.io import loadmat
from scipy.signal import windows
from scipy.fft import fft, fftfreq


# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="IEEE34 – V8 (Play estável no Streamlit + Sequências + RMS único)",
    layout="wide"
)

ANGLES = np.deg2rad([0, 120, 240])  # A,B,C

# Tema dark
PAPER_BG = "#0f1117"
PLOT_BG  = "#0f1117"
GRID_CLR = "rgba(255,255,255,0.08)"
ZERO_CLR = "rgba(255,255,255,0.10)"
FONT_CLR = "#e6edf3"
LEG_CLR  = "#c9d1d9"

# Cores
CLR_A = "#FF5555"
CLR_B = "#55FF55"
CLR_C = "#5555FF"
CLR_RES = "rgba(240,246,252,0.95)"

CLR_V1 = "rgba(88,166,255,0.95)"
CLR_V2 = "rgba(170,170,170,0.95)"
CLR_V0 = "rgba(255,215,0,0.95)"

CLR_TRAJ_ABC = "rgba(240,246,252,0.35)"
CLR_TRAJ_V1  = "rgba(88,166,255,0.35)"
CLR_TRAJ_V2  = "rgba(170,170,170,0.35)"
CLR_TRAJ_V0  = "rgba(255,215,0,0.35)"


# =========================================================
# CORE MATH: Clarke, FFT, THD
# =========================================================
def clarke_transform(a, b, c, mode="power"):
    k = (2/3) if mode == "amp" else np.sqrt(2/3)
    alpha = k * (a - 0.5*b - 0.5*c)
    beta  = k * ((np.sqrt(3)/2)*b - (np.sqrt(3)/2)*c)
    return alpha, beta


def compute_fft_rms(signal, fs, window="hann", remove_mean=True):
    signal = np.asarray(signal).squeeze()
    N = len(signal)
    if N < 4 or fs <= 0:
        return np.array([]), np.array([])

    x = signal.copy()
    if remove_mean:
        x = x - np.mean(x)

    w = windows.hann(N) if window == "hann" else np.ones(N)
    X = fft(x * w)
    freqs = fftfreq(N, d=1/fs)

    pos = freqs >= 0
    freqs = freqs[pos]
    X = X[pos]

    X_mag = (2.0 / np.sum(w)) * np.abs(X)
    X_rms = X_mag / np.sqrt(2)
    return freqs, X_rms


def compute_thd_percent(freqs, spectrum_rms, f_fund=60.0, h_max=25, tol_hz=1.0):
    if len(freqs) == 0:
        return np.nan

    freqs = np.asarray(freqs)
    spec = np.asarray(spectrum_rms)

    idx = np.where(np.abs(freqs - f_fund) <= tol_hz)[0]
    if len(idx) == 0:
        return np.nan
    X1 = spec[idx[0]]
    if X1 <= 1e-12:
        return np.nan

    harm_sq = 0.0
    for h in range(2, int(h_max) + 1):
        fh = h * f_fund
        ih = np.where(np.abs(freqs - fh) <= tol_hz)[0]
        if len(ih) > 0:
            harm_sq += spec[ih[0]]**2

    return float(np.sqrt(harm_sq) / X1 * 100.0)


# =========================================================
# SEQUÊNCIAS SIMÉTRICAS (Fortescue)
# =========================================================
def phasor_at_f0(x, t, f0=60.0, window="hann", remove_mean=True):
    x = np.asarray(x).squeeze()
    t = np.asarray(t).squeeze()
    if len(x) != len(t) or len(x) < 8:
        return 0.0 + 0.0j

    if remove_mean:
        x = x - np.mean(x)

    N = len(x)
    w = windows.hann(N) if window == "hann" else np.ones(N)
    xw = x * w

    exp_term = np.exp(-1j * 2*np.pi*f0*t)
    X = np.sum(xw * exp_term)
    W = np.sum(w)

    A_peak = 2.0 * X / W
    Vrms = A_peak / np.sqrt(2)
    return Vrms


def symmetrical_components(Va, Vb, Vc):
    a = np.exp(1j * 2*np.pi/3)
    T = (1/3) * np.array([
        [1,   1,   1],
        [1, a**2,  a],
        [1,   a, a**2]
    ], dtype=complex)
    V0, V1, V2 = T @ np.array([Va, Vb, Vc], dtype=complex)
    return V0, V1, V2


def inv_symmetrical_components(V0, V1, V2):
    a = np.exp(1j * 2*np.pi/3)
    Ti = np.array([
        [1,   1,   1],
        [1,   a, a**2],
        [1, a**2,  a]
    ], dtype=complex)
    Va, Vb, Vc = Ti @ np.array([V0, V1, V2], dtype=complex)
    return Va, Vb, Vc


def synth_from_phasor(Vrms, t, f0):
    return np.sqrt(2) * np.real(Vrms * np.exp(1j * 2*np.pi*f0*t))


# =========================================================
# MATLAB EXTRACTION
# =========================================================
def _mat_getfield(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    try:
        if hasattr(obj, "dtype") and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    except Exception:
        pass
    return None


def extract_ts_from_mat(mat, point_name):
    if 'ts' not in mat:
        return None, None

    ts_root = mat['ts']
    if isinstance(ts_root, np.ndarray):
        ts_root = ts_root.squeeze()
        if isinstance(ts_root, np.ndarray) and ts_root.dtype == object:
            ts_root = ts_root.item()

    key = f"ts_{point_name}"
    entry = _mat_getfield(ts_root, key)
    if entry is None:
        return None, None

    if isinstance(entry, np.ndarray):
        entry = entry.squeeze()
        if isinstance(entry, np.ndarray) and entry.dtype == object:
            entry = entry.item()

    time_ = _mat_getfield(entry, "Time")
    data_ = _mat_getfield(entry, "Data")
    if time_ is None or data_ is None:
        return None, None

    t = np.asarray(time_).squeeze()
    x = np.asarray(data_).squeeze()

    if x.ndim == 2 and x.shape[0] == 3 and x.shape[1] == t.shape[0]:
        x = x.T

    return t, x


def extract_m1_location(mat):
    if 'm1_location' not in mat:
        return None
    try:
        return float(np.asarray(mat['m1_location']).squeeze())
    except Exception:
        return None


# =========================================================
# Campos XY
# =========================================================
def resultant_xy_series(a, b, c):
    xa = a * np.cos(ANGLES[0]); ya = a * np.sin(ANGLES[0])
    xb = b * np.cos(ANGLES[1]); yb = b * np.sin(ANGLES[1])
    xc = c * np.cos(ANGLES[2]); yc = c * np.sin(ANGLES[2])
    rx = xa + xb + xc
    ry = ya + yb + yc
    return rx, ry


def abc_vectors_xy(a_val, b_val, c_val):
    vals = np.array([a_val, b_val, c_val], dtype=float)
    xs = vals * np.cos(ANGLES)
    ys = vals * np.sin(ANGLES)
    vecs = list(zip(xs, ys))
    rx = sum(v[0] for v in vecs)
    ry = sum(v[1] for v in vecs)
    return vecs, (rx, ry)


# =========================================================
# FIGURA (sem frames do Plotly) – atualiza por idx
# =========================================================
def build_static_figure(t, a, b, c, alpha, beta, seq_data, idx, traj_stride=3, show_seq=True, clarke_label="power"):
    N = len(t)
    idx = int(np.clip(idx, 0, N - 1))

    rxt, ryt = resultant_xy_series(a, b, c)
    rx1, ry1 = resultant_xy_series(seq_data["a1"], seq_data["b1"], seq_data["c1"])
    rx2, ry2 = resultant_xy_series(seq_data["a2"], seq_data["b2"], seq_data["c2"])
    rx0, ry0 = resultant_xy_series(seq_data["a0"], seq_data["b0"], seq_data["c0"])

    vmax = float(np.max(np.abs(np.vstack([a, b, c]))))
    axis_lim = max(1.0, 2.8 * vmax)

    ab_lim = max(1.0, 1.1 * float(np.max(np.sqrt(alpha**2 + beta**2))))

    seq_lim = max(1.0, 1.2 * float(np.max(np.abs(np.vstack([rx1, ry1, rx2, ry2, rx0, ry0])))))

    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            "ABC (tempo)",
            "Campo ABC (XY)",
            f"Clarke αβ (tempo) – {clarke_label}",
            "Plano αβ",
            "Campo V1 (sequência positiva)",
            "Campo V2 (sequência negativa)",
            "Campo V0 (sequência zero)",
            "—"
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.10,
    )

    # ABC tempo (com marcador idx)
    fig.add_trace(go.Scatter(x=t, y=a, mode="lines", name="A", line=dict(color=CLR_A, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=b, mode="lines", name="B", line=dict(color=CLR_B, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=c, mode="lines", name="C", line=dict(color=CLR_C, width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[idx]], y=[a[idx]], mode="markers", name="A@t", marker=dict(size=9, color=CLR_A)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[idx]], y=[b[idx]], mode="markers", name="B@t", marker=dict(size=9, color=CLR_B)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[idx]], y=[c[idx]], mode="markers", name="C@t", marker=dict(size=9, color=CLR_C)), row=1, col=1)

    # Campo ABC (vetores + resultante + rastro)
    vecs, _ = abc_vectors_xy(a[idx], b[idx], c[idx])
    for (vx, vy), colr, nm in zip(vecs, [CLR_A, CLR_B, CLR_C], ["A", "B", "C"]):
        fig.add_trace(go.Scatter(
            x=[0, vx], y=[0, vy], mode="lines+markers", name=nm,
            line=dict(width=4, color=colr), marker=dict(size=7, color=colr)
        ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[0, rxt[idx]], y=[0, ryt[idx]],
        mode="lines+markers", name="Resultante",
        line=dict(width=5, color=CLR_RES), marker=dict(size=9, color=CLR_RES)
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=rxt[:idx+1:traj_stride], y=ryt[:idx+1:traj_stride],
        mode="lines", name="rastro ABC", line=dict(width=2, color=CLR_TRAJ_ABC)
    ), row=1, col=2)

    # Clarke tempo + marcador
    fig.add_trace(go.Scatter(x=t, y=alpha, mode="lines", name="α", line=dict(width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=beta,  mode="lines", name="β", line=dict(width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[t[idx]], y=[alpha[idx]], mode="markers", name="α@t", marker=dict(size=8)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[t[idx]], y=[beta[idx]],  mode="markers", name="β@t", marker=dict(size=8)), row=2, col=1)

    # Plano αβ (traj completa + ponto + vetor + rastro até idx)
    fig.add_trace(go.Scatter(x=alpha, y=beta, mode="lines", name="traj αβ", line=dict(width=2)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="origem", marker=dict(size=7)), row=2, col=2)
    fig.add_trace(go.Scatter(x=alpha[:idx+1:traj_stride], y=beta[:idx+1:traj_stride], mode="lines",
                             name="rastro αβ", line=dict(width=2, color="rgba(255,255,255,0.25)")), row=2, col=2)
    fig.add_trace(go.Scatter(x=[alpha[idx]], y=[beta[idx]], mode="markers", name="ponto αβ", marker=dict(size=10)), row=2, col=2)
    fig.add_trace(go.Scatter(x=[0, alpha[idx]], y=[0, beta[idx]], mode="lines", name="vetor αβ",
                             line=dict(width=3, dash="dot", color="rgba(255,215,0,0.9)")), row=2, col=2)

    # Campos V1/V2/V0 com rastros
    rx1, ry1 = resultant_xy_series(seq_data["a1"], seq_data["b1"], seq_data["c1"])
    rx2, ry2 = resultant_xy_series(seq_data["a2"], seq_data["b2"], seq_data["c2"])
    rx0, ry0 = resultant_xy_series(seq_data["a0"], seq_data["b0"], seq_data["c0"])

    def add_seq_field(row, col, name_vec, name_traj, color_vec, color_traj, rx, ry):
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="origem", marker=dict(size=7)), row=row, col=col)
        fig.add_trace(go.Scatter(x=rx[:idx+1:traj_stride], y=ry[:idx+1:traj_stride], mode="lines",
                                 name=name_traj, line=dict(width=2, color=color_traj)), row=row, col=col)
        fig.add_trace(go.Scatter(x=[0, rx[idx]], y=[0, ry[idx]], mode="lines+markers",
                                 name=name_vec, line=dict(width=4, color=color_vec),
                                 marker=dict(size=8, color=color_vec)), row=row, col=col)

    if show_seq:
        add_seq_field(3, 1, "V1 (vetor)", "rastro V1", CLR_V1, CLR_TRAJ_V1, rx1, ry1)
        add_seq_field(3, 2, "V2 (vetor)", "rastro V2", CLR_V2, CLR_TRAJ_V2, rx2, ry2)
        add_seq_field(4, 1, "V0 (vetor)", "rastro V0", CLR_V0, CLR_TRAJ_V0, rx0, ry0)
    else:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="seq off"), row=3, col=1)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="seq off"), row=3, col=2)
        fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="seq off"), row=4, col=1)

    # Layout
    fig.update_layout(
        height=1280,
        margin=dict(l=20, r=20, t=85, b=40),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR, size=13),
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom", font=dict(color=LEG_CLR)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)

    # Eixos
    fig.update_xaxes(title_text="Tempo (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    fig.update_xaxes(range=[-axis_lim, axis_lim], title_text="X", row=1, col=2)
    fig.update_yaxes(range=[-axis_lim, axis_lim], title_text="Y", row=1, col=2)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

    fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    fig.update_xaxes(range=[-ab_lim, ab_lim], title_text="α", row=2, col=2)
    fig.update_yaxes(range=[-ab_lim, ab_lim], title_text="β", row=2, col=2)
    fig.update_yaxes(scaleanchor="x4", scaleratio=1, row=2, col=2)

    fig.update_xaxes(range=[-seq_lim, seq_lim], title_text="X", row=3, col=1)
    fig.update_yaxes(range=[-seq_lim, seq_lim], title_text="Y", row=3, col=1)
    fig.update_yaxes(scaleanchor="x5", scaleratio=1, row=3, col=1)

    fig.update_xaxes(range=[-seq_lim, seq_lim], title_text="X", row=3, col=2)
    fig.update_yaxes(range=[-seq_lim, seq_lim], title_text="Y", row=3, col=2)
    fig.update_yaxes(scaleanchor="x6", scaleratio=1, row=3, col=2)

    fig.update_xaxes(range=[-seq_lim, seq_lim], title_text="X", row=4, col=1)
    fig.update_yaxes(range=[-seq_lim, seq_lim], title_text="Y", row=4, col=1)
    fig.update_yaxes(scaleanchor="x7", scaleratio=1, row=4, col=1)

    return fig


def rms_sequences_figure(V0, V1, V2):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["V1 (positiva)", "V2 (negativa)", "V0 (zero)"],
        y=[float(np.abs(V1)), float(np.abs(V2)), float(np.abs(V0))],
        marker=dict(color=[CLR_V1, CLR_V2, CLR_V0])
    ))
    fig.update_layout(
        title="RMS das Componentes Simétricas (|V1|, |V2|, |V0|)",
        height=320,
        margin=dict(l=20, r=20, t=55, b=40),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR),
    )
    fig.update_yaxes(title_text="RMS", showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)
    fig.update_xaxes(showgrid=False)
    return fig


# =========================================================
# STATE (Play estável)
# =========================================================
def ensure_state():
    if "playing" not in st.session_state:
        st.session_state.playing = False
    if "idx" not in st.session_state:
        st.session_state.idx = 0

ensure_state()


# =========================================================
# UI
# =========================================================
st.markdown("## ⚡ IEEE 34 Barras — V8 (Play estável no Streamlit)")
st.caption("Aqui o Play NÃO depende do Plotly. Ele atualiza por rerun, então o gráfico não some.")


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Controles")

    st.markdown("### 📂 Arquivos .mat")
    uploaded = st.file_uploader("Envie um ou vários .mat", type=["mat"], accept_multiple_files=True)
    if not uploaded:
        st.info("Envie arquivos .mat para começar.")
        st.stop()

    mats = {}
    errors = []
    for uf in uploaded:
        try:
            mats[uf.name] = loadmat(io.BytesIO(uf.getvalue()), squeeze_me=False, struct_as_record=False)
        except Exception as e:
            errors.append((uf.name, str(e)))

    if errors:
        st.warning("Alguns arquivos falharam ao carregar:")
        for name, err in errors:
            st.write(f"- {name}: {err}")

    if not mats:
        st.error("Nenhum arquivo válido foi carregado.")
        st.stop()

    file_names = sorted(mats.keys())
    selected_file = st.selectbox("Arquivo ativo", file_names)
    mat = mats[selected_file]

    st.divider()
    st.markdown("### 📍 Sinal")
    points = ['I_800','V_800','I_T2F','V_T2F','I_818','V_818','I_820','V_820','I_822','V_822']
    point = st.selectbox("Ponto", points, index=0)

    remove_mean = st.checkbox("Remover offset (DC)", value=True)

    st.divider()
    st.markdown("### 🔁 Clarke")
    clarke_mode = st.radio("Modo", ["Power Invariant (√2/3)", "Amplitude Invariant (2/3)"], horizontal=True)
    clarke_mode_key = "power" if "Power" in clarke_mode else "amp"

    st.divider()
    st.markdown("### 🧩 Componentes Simétricas")
    show_seq = st.checkbox("Mostrar campos V0/V1/V2", value=True)
    seq_f0 = st.number_input("f0 p/ fasor (Hz)", value=60.0, min_value=1.0, step=1.0)
    seq_window = st.selectbox("Janela p/ fasor", ["hann", "rect"], index=0)

    st.divider()
    st.markdown("### 🎞️ Play (estável)")
    speed_ms = st.slider("Velocidade (ms)", 20, 250, 60, 5)
    step = st.slider("Passo por tick (amostras)", 1, 200, 5, 1)
    traj_stride = st.slider("Densidade do rastro (traj_stride)", 1, 20, 3, 1)

    st.divider()
    st.markdown("### 📊 FFT / THD")
    show_fft = st.checkbox("Ativar FFT/THD", value=False)
    if show_fft:
        window_type = st.selectbox("Janela FFT", ["hann", "rect"], index=0)
        f_fund = st.number_input("Fundamental THD (Hz)", value=60.0, min_value=1.0, step=1.0)
        h_max = st.slider("Máx harmônica THD", 5, 60, 25, 1)
        tol_hz = st.number_input("Tolerância THD (Hz)", value=1.0, min_value=0.1, step=0.1)
        fft_xmax = st.number_input("Limite X FFT (Hz)", value=float(h_max)*float(f_fund) + 10.0, min_value=10.0, step=10.0)


# =========================================================
# LOAD SIGNAL
# =========================================================
m1 = extract_m1_location(mat)
t, x = extract_ts_from_mat(mat, point)

if t is None or x is None:
    st.error(f"Não encontrei `ts_{point}.Time/Data` no arquivo `{selected_file}`.")
    st.stop()

t = np.asarray(t).squeeze()

if not (isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] >= 3):
    st.error(f"Esta versão exige sinal trifásico N×3. Recebi x.shape={getattr(x, 'shape', None)}.")
    st.stop()

a, b, c = x[:, 0], x[:, 1], x[:, 2]
if remove_mean:
    a = a - np.mean(a); b = b - np.mean(b); c = c - np.mean(c)

dt = np.diff(t)
fs = 1.0 / float(np.mean(dt)) if len(dt) > 0 else 0.0

alpha, beta = clarke_transform(a, b, c, mode=clarke_mode_key)


# =========================================================
# SEQUÊNCIAS
# =========================================================
Va = phasor_at_f0(a, t, f0=float(seq_f0), window=seq_window, remove_mean=False)
Vb = phasor_at_f0(b, t, f0=float(seq_f0), window=seq_window, remove_mean=False)
Vc = phasor_at_f0(c, t, f0=float(seq_f0), window=seq_window, remove_mean=False)

V0, V1, V2 = symmetrical_components(Va, Vb, Vc)

Va0, Vb0, Vc0 = inv_symmetrical_components(V0, 0, 0)
Va1, Vb1, Vc1 = inv_symmetrical_components(0, V1, 0)
Va2, Vb2, Vc2 = inv_symmetrical_components(0, 0, V2)

a0 = synth_from_phasor(Va0, t, seq_f0); b0 = synth_from_phasor(Vb0, t, seq_f0); c0 = synth_from_phasor(Vc0, t, seq_f0)
a1 = synth_from_phasor(Va1, t, seq_f0); b1 = synth_from_phasor(Vb1, t, seq_f0); c1 = synth_from_phasor(Vc1, t, seq_f0)
a2 = synth_from_phasor(Va2, t, seq_f0); b2 = synth_from_phasor(Vb2, t, seq_f0); c2 = synth_from_phasor(Vc2, t, seq_f0)

seq_data = {"a0": a0, "b0": b0, "c0": c0, "a1": a1, "b1": b1, "c1": c1, "a2": a2, "b2": b2, "c2": c2}


# =========================================================
# CONTROLES PRINCIPAIS (PLAY ESTÁVEL)
# =========================================================
N = len(t)
st.session_state.idx = int(np.clip(st.session_state.idx, 0, N-1))

top1, top2, top3, top4, top5 = st.columns([1.2, 1.2, 1.0, 1.0, 2.0])

with top1:
    if st.button("▶ Play" if not st.session_state.playing else "⏸ Pause", use_container_width=True):
        st.session_state.playing = not st.session_state.playing

with top2:
    if st.button("⟲ Reset", use_container_width=True):
        st.session_state.playing = False
        st.session_state.idx = 0

with top3:
    if st.button("⏭ Step +", use_container_width=True):
        st.session_state.playing = False
        st.session_state.idx = min(N-1, st.session_state.idx + step)

with top4:
    if st.button("⏮ Step -", use_container_width=True):
        st.session_state.playing = False
        st.session_state.idx = max(0, st.session_state.idx - step)

with top5:
    st.session_state.idx = st.slider("Frame (índice)", 0, N-1, st.session_state.idx, 1)

st.caption(f"Arquivo: `{selected_file}` | Ponto: `{point}` | m1: `{m1}` | fs ~ {fs:.2f} Hz | t = {t[st.session_state.idx]:.6f} s")


# =========================================================
# RMS V0/V1/V2 (JUNTOS)
# =========================================================
st.markdown("### 📌 Componentes Simétricas (RMS) – V0/V1/V2 (juntos)")
cA, cB = st.columns([1.1, 1.0], gap="large")

with cA:
    st.plotly_chart(rms_sequences_figure(V0, V1, V2), use_container_width=True)

with cB:
    st.table({
        "Componente": ["V1 (Positiva)", "V2 (Negativa)", "V0 (Zero)"],
        "|V| RMS": [float(np.abs(V1)), float(np.abs(V2)), float(np.abs(V0))],
        "∠ (deg)": [float(np.angle(V1, deg=True)), float(np.angle(V2, deg=True)), float(np.angle(V0, deg=True))]
    })


# =========================================================
# FIGURA PRINCIPAL (atualiza por idx)
# =========================================================
fig = build_static_figure(
    t=t, a=a, b=b, c=c,
    alpha=alpha, beta=beta,
    seq_data=seq_data,
    idx=st.session_state.idx,
    traj_stride=traj_stride,
    show_seq=show_seq,
    clarke_label=("power" if clarke_mode_key == "power" else "amp")
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# FFT/THD (optional)
# =========================================================
if show_fft:
    st.divider()
    st.markdown("## 📊 FFT e THD")

    if fs <= 0 or len(a) < 8:
        st.warning("FFT/THD ativados, mas fs inválida ou sinal curto.")
    else:
        freqs, Fa = compute_fft_rms(a, fs, window=window_type, remove_mean=False)
        _, Fb = compute_fft_rms(b, fs, window=window_type, remove_mean=False)
        _, Fc = compute_fft_rms(c, fs, window=window_type, remove_mean=False)

        thd_a = compute_thd_percent(freqs, Fa, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)
        thd_b = compute_thd_percent(freqs, Fb, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)
        thd_c = compute_thd_percent(freqs, Fc, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)

        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("THD (%) por fase")
            st.table({
                "Fase": ["A", "B", "C"],
                "THD (%)": [
                    None if np.isnan(thd_a) else round(thd_a, 2),
                    None if np.isnan(thd_b) else round(thd_b, 2),
                    None if np.isnan(thd_c) else round(thd_c, 2),
                ]
            })

        with c2:
            fig_fft = go.Figure()
            fig_fft.add_trace(go.Scatter(x=freqs, y=Fa, mode="lines", name="A", line=dict(color=CLR_A)))
            fig_fft.add_trace(go.Scatter(x=freqs, y=Fb, mode="lines", name="B", line=dict(color=CLR_B)))
            fig_fft.add_trace(go.Scatter(x=freqs, y=Fc, mode="lines", name="C", line=dict(color=CLR_C)))
            fig_fft.update_layout(
                title=f"FFT RMS — janela {window_type}",
                xaxis_title="Frequência (Hz)",
                yaxis_title="Amplitude (RMS)",
                height=380,
                margin=dict(l=25, r=15, t=60, b=35),
                paper_bgcolor=PAPER_BG,
                plot_bgcolor=PLOT_BG,
                font=dict(color=FONT_CLR),
                legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
            )
            fig_fft.update_xaxes(range=[0, float(fft_xmax)], showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)
            fig_fft.update_yaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)
            st.plotly_chart(fig_fft, use_container_width=True)


# =========================================================
# LOOP PLAY (Streamlit rerun)
# =========================================================
if st.session_state.playing:
    time.sleep(speed_ms / 1000.0)
    nxt = st.session_state.idx + step
    if nxt >= N:
        nxt = 0
    st.session_state.idx = nxt
    st.rerun()
