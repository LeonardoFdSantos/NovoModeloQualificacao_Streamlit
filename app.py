import io
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
    page_title="IEEE34 – V4 (UI + Multi .mat + Animação + Botões Dark)",
    layout="wide"
)

ANGLES = np.deg2rad([0, 120, 240])  # A,B,C

# Tema dark (ajuste se quiser)
PAPER_BG = "#0f1117"
PLOT_BG  = "#0f1117"
GRID_CLR = "rgba(255,255,255,0.08)"
ZERO_CLR = "rgba(255,255,255,0.10)"
FONT_CLR = "#e6edf3"
LEG_CLR  = "#c9d1d9"

BTN_BG   = "rgba(22,27,34,0.95)"
BTN_BRD  = "rgba(88,166,255,0.55)"

SL_BG    = "rgba(22,27,34,0.75)"
SL_BRD   = "rgba(240,246,252,0.25)"


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

    # magnitude unilateral corrigida + RMS
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
# MATLAB EXTRACTION (robust-ish)
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
    """
    Espera:
      mat['ts'] -> struct
      ts.ts_<POINT>.Time / .Data
    """
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

    # se vier (3,N), transpõe p/ (N,3)
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
# PHASOR HELPERS
# =========================================================
def abc_vectors_xy(a_val, b_val, c_val):
    vals = np.array([a_val, b_val, c_val], dtype=float)
    xs = vals * np.cos(ANGLES)
    ys = vals * np.sin(ANGLES)
    vecs = list(zip(xs, ys))
    rx = sum(v[0] for v in vecs)
    ry = sum(v[1] for v in vecs)
    return vecs, (rx, ry)


# =========================================================
# ANIMATED FIGURE (slider limpo + botões dark)
# =========================================================
def build_animated_figure(t, a, b, c, alpha, beta, frame_step=5, clarke_label="power"):
    N = len(t)

    # Frames usados na animação
    frame_idxs = list(range(0, N, frame_step))
    if frame_idxs[-1] != N - 1:
        frame_idxs.append(N - 1)

    # Slider com menos steps (evita poluição)
    max_slider_steps = 60
    if len(frame_idxs) > max_slider_steps:
        slider_idxs = np.linspace(0, len(frame_idxs) - 1, max_slider_steps).astype(int)
        slider_frame_idxs = [frame_idxs[i] for i in slider_idxs]
    else:
        slider_frame_idxs = frame_idxs

    vmax = float(np.max(np.abs([a, b, c])))
    axis_lim = max(1.0, 2.8 * vmax)

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "ABC (tempo)",
            "Campo ABC (XY)",
            f"Clarke αβ (tempo) – {clarke_label}",
            "Plano αβ"
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    # -------------------------
    # Traces estáticos
    # -------------------------
    fig.add_trace(go.Scatter(x=t, y=a, mode="lines", name="A"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=b, mode="lines", name="B"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=c, mode="lines", name="C"), row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=alpha, mode="lines", name="α"), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=beta,  mode="lines", name="β"), row=2, col=1)

    fig.add_trace(
        go.Scatter(x=alpha, y=beta, mode="lines", name="traj αβ", line=dict(width=2)),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=[0], y=[0], mode="markers", name="origem", marker=dict(size=8)),
        row=2, col=2
    )

    # -------------------------
    # Traces dinâmicos (primeiro frame)
    # -------------------------
    i0 = frame_idxs[0]

    # Marcadores ABC
    fig.add_trace(go.Scatter(x=[t[i0]], y=[a[i0]], mode="markers", name="A@t", marker=dict(size=10)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[i0]], y=[b[i0]], mode="markers", name="B@t", marker=dict(size=10)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[i0]], y=[c[i0]], mode="markers", name="C@t", marker=dict(size=10)),
                  row=1, col=1)

    # Campo ABC
    vecs0, res0 = abc_vectors_xy(a[i0], b[i0], c[i0])
    colors = ["#FF5555", "#55FF55", "#5555FF"]
    names = ["A", "B", "C"]

    for (vx, vy), colr, nm in zip(vecs0, colors, names):
        fig.add_trace(go.Scatter(
            x=[0, vx], y=[0, vy],
            mode="lines+markers", name=nm,
            line=dict(width=4, color=colr),
            marker=dict(size=8)
        ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=[0, res0[0]], y=[0, res0[1]],
        mode="lines+markers", name="Resultante",
        line=dict(width=5, color="white"),
        marker=dict(size=10)
    ), row=1, col=2)

    # Marcadores αβ no tempo
    fig.add_trace(go.Scatter(x=[t[i0]], y=[alpha[i0]], mode="markers", name="α@t", marker=dict(size=10)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=[t[i0]], y=[beta[i0]],  mode="markers", name="β@t", marker=dict(size=10)),
                  row=2, col=1)

    # Plano αβ (ponto + vetor)
    fig.add_trace(go.Scatter(x=[alpha[i0]], y=[beta[i0]], mode="markers", name="ponto αβ", marker=dict(size=12)),
                  row=2, col=2)
    fig.add_trace(go.Scatter(x=[0, alpha[i0]], y=[0, beta[i0]], mode="lines", name="vetor αβ",
                             line=dict(width=3, dash="dot")),
                  row=2, col=2)

    # Índices dos traces dinâmicos
    dynamic_trace_idxs = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

    # -------------------------
    # Frames
    # -------------------------
    frames = []
    for k in frame_idxs:
        vecs, res = abc_vectors_xy(a[k], b[k], c[k])

        frame_data = []
        # ABC markers
        frame_data.append(go.Scatter(x=[t[k]], y=[a[k]]))
        frame_data.append(go.Scatter(x=[t[k]], y=[b[k]]))
        frame_data.append(go.Scatter(x=[t[k]], y=[c[k]]))

        # phasor A,B,C
        for (vx, vy) in vecs:
            frame_data.append(go.Scatter(x=[0, vx], y=[0, vy]))

        # resultant
        frame_data.append(go.Scatter(x=[0, res[0]], y=[0, res[1]]))

        # clarke markers
        frame_data.append(go.Scatter(x=[t[k]], y=[alpha[k]]))
        frame_data.append(go.Scatter(x=[t[k]], y=[beta[k]]))

        # plane marker + vector
        frame_data.append(go.Scatter(x=[alpha[k]], y=[beta[k]]))
        frame_data.append(go.Scatter(x=[0, alpha[k]], y=[0, beta[k]]))

        frames.append(go.Frame(data=frame_data, name=str(k), traces=dynamic_trace_idxs))

    fig.frames = frames

    # -------------------------
    # Slider limpo (sem labels poluindo)
    # -------------------------
    slider_steps = []
    for k in slider_frame_idxs:
        slider_steps.append(dict(
            method="animate",
            args=[[str(k)], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0}
            }],
            label=""  # remove textos 0.461s, 0.492s...
        ))

    # -------------------------
    # Layout (dark) + botões bonitos
    # -------------------------
    fig.update_layout(
        height=860,
        margin=dict(l=20, r=20, t=90, b=120),

        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR, size=13),

        legend=dict(
            orientation="h",
            y=1.02, x=1,
            xanchor="right", yanchor="bottom",
            font=dict(color=LEG_CLR)
        ),

        # Botões (Play/Pause) com fundo escuro e borda azul
        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.0, y=1.18,
            xanchor="left",
            yanchor="top",
            showactive=True,
            active=0,

            bgcolor=BTN_BG,
            bordercolor=BTN_BRD,
            borderwidth=1,
            pad=dict(r=8, t=6, l=8, b=6),

            font=dict(color=FONT_CLR, size=14),

            buttons=[
                dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 0, "redraw": False},
                        "transition": {"duration": 0},
                        "fromcurrent": True,
                        "mode": "immediate"
                    }]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "transition": {"duration": 0},
                        "mode": "immediate"
                    }]
                ),
            ]
        )],

        # Slider abaixo do grid + estilo dark
        sliders=[dict(
            x=0.0, y=-0.10, len=1.0,
            pad=dict(t=10, b=0),
            currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=14, color=FONT_CLR), visible=True),
            bgcolor=SL_BG,
            bordercolor=SL_BRD,
            borderwidth=1,
            steps=slider_steps
        )],
    )

    # Grid/zero-line para tema dark
    fig.update_xaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)

    # Axes titles/ranges
    fig.update_xaxes(title_text="Tempo (s)", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)

    fig.update_xaxes(range=[-axis_lim, axis_lim], row=1, col=2)
    fig.update_yaxes(range=[-axis_lim, axis_lim], row=1, col=2)
    fig.update_yaxes(scaleanchor="x2", scaleratio=1, row=1, col=2)

    fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    fig.update_xaxes(title_text="α", row=2, col=2)
    fig.update_yaxes(title_text="β", row=2, col=2)
    fig.update_yaxes(scaleanchor="x4", scaleratio=1, row=2, col=2)

    return fig


# =========================================================
# UI: HEADER
# =========================================================
st.markdown("## ⚡ IEEE 34 Barras — Visualização Interativa")
st.caption("Upload múltiplo • Seleção do arquivo • Animação fluida • Botões/slider dark • Clarke • FFT/THD opcional")


# =========================================================
# SIDEBAR: UI + MULTI-ARQUIVO
# =========================================================
with st.sidebar:
    st.markdown("## ⚙️ Controles")

    st.markdown("### 📂 Arquivos .mat")
    uploaded = st.file_uploader(
        "Envie um ou vários arquivos .mat",
        type=["mat"],
        accept_multiple_files=True
    )

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
    selected_file = st.selectbox(
        "Arquivo ativo",
        file_names,
        format_func=lambda x: f"📄 {x}"
    )
    mat = mats[selected_file]

    st.divider()

    st.markdown("### 📍 Sinal")
    points = [
        'I_800','V_800','I_T2F','V_T2F',
        'I_818','V_818','I_820','V_820','I_822','V_822'
    ]
    point = st.selectbox("Ponto", points, index=0)

    remove_mean = st.checkbox("Remover offset (DC)", value=True)

    clarke_mode = st.radio(
        "Clarke",
        ["Power Invariant (√2/3)", "Amplitude Invariant (2/3)"],
        horizontal=True
    )
    clarke_mode_key = "power" if "Power" in clarke_mode else "amp"

    st.divider()

    st.markdown("### 🎞️ Animação")
    frame_step = st.slider("frame_step (↓ mais suave)", 1, 50, 5, 1)
    st.caption("Sugestão: 5 a 10 (equilíbrio entre suavidade e leveza)")

    st.divider()

    st.markdown("### 📊 FFT / THD")
    show_fft = st.checkbox("Ativar FFT/THD", value=False)
    if show_fft:
        window_type = st.selectbox("Janela", ["hann", "rect"], index=0)
        f_fund = st.number_input("Fundamental (Hz)", value=60.0, min_value=1.0, step=1.0)
        h_max = st.slider("Máx harmônica THD", 5, 60, 25, 1)
        tol_hz = st.number_input("Tolerância (Hz)", value=1.0, min_value=0.1, step=0.1)
        fft_xmax = st.number_input("Limite X FFT (Hz)", value=float(h_max)*float(f_fund) + 10.0, min_value=10.0, step=10.0)


# =========================================================
# LOAD & VALIDATE SIGNAL
# =========================================================
m1 = extract_m1_location(mat)
t, x = extract_ts_from_mat(mat, point)

if t is None or x is None:
    st.error(f"Não encontrei `ts_{point}.Time/Data` dentro de `ts` no arquivo `{selected_file}`.")
    st.stop()

t = np.asarray(t).squeeze()

if not (isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] >= 3):
    st.error(
        f"Para a animação fluida (Plotly frames), o ponto precisa ser trifásico N×3.\n\n"
        f"Recebi x.shape={getattr(x, 'shape', None)} no arquivo `{selected_file}`."
    )
    st.stop()

a, b, c = x[:, 0], x[:, 1], x[:, 2]

if remove_mean:
    a = a - np.mean(a)
    b = b - np.mean(b)
    c = c - np.mean(c)

dt = np.diff(t)
fs = 1.0 / float(np.mean(dt)) if len(dt) > 0 else 0.0

alpha, beta = clarke_transform(a, b, c, mode=clarke_mode_key)


# =========================================================
# MAIN VIEW
# =========================================================
st.markdown(
    f"### 🧪 Caso selecionado\n"
    f"- **Arquivo:** `{selected_file}`\n"
    f"- **Ponto:** `{point}`\n"
    f"- **m1:** `{m1}`\n"
    f"- **fs estimada:** `{fs:.2f} Hz`\n"
    f"- **N:** `{len(t)}` | **frame_step:** `{frame_step}`"
)

fig = build_animated_figure(
    t=t, a=a, b=b, c=c,
    alpha=alpha, beta=beta,
    frame_step=frame_step,
    clarke_label=("power" if clarke_mode_key == "power" else "amp")
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# FFT / THD (optional)
# =========================================================
if show_fft:
    if fs <= 0 or len(a) < 8:
        st.warning("FFT/THD ativados, mas fs inválida ou sinal curto.")
    else:
        freqs, Fa = compute_fft_rms(a, fs, window=window_type, remove_mean=False)
        _, Fb = compute_fft_rms(b, fs, window=window_type, remove_mean=False)
        _, Fc = compute_fft_rms(c, fs, window=window_type, remove_mean=False)

        thd_a = compute_thd_percent(freqs, Fa, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)
        thd_b = compute_thd_percent(freqs, Fb, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)
        thd_c = compute_thd_percent(freqs, Fc, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)

        st.divider()
        st.markdown("## 📊 FFT e THD")

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
            fig_fft.add_trace(go.Scatter(x=freqs, y=Fa, mode="lines", name="A"))
            fig_fft.add_trace(go.Scatter(x=freqs, y=Fb, mode="lines", name="B"))
            fig_fft.add_trace(go.Scatter(x=freqs, y=Fc, mode="lines", name="C"))
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
