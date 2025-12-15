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
    page_title="IEEE34 – V7 (Campos + Rastros + Sequências V0/V1/V2)",
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

# Botões plotly
BTN_BG   = "rgba(22,27,34,0.95)"
BTN_BRD  = "rgba(88,166,255,0.55)"

# Slider plotly
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
    """
    Estima fasor complexo RMS na frequência f0 via projeção (DFT na freq alvo).
    """
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

    # Sinal real: A_peak ~ 2X/W
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

    # se vier (3,N), transpõe para (N,3)
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
# Campo ABC: resultante XY por amostra
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
# FIGURA ANIMADA: 3x2 + CAPTIONS
# =========================================================
def build_animated_figure(
    t, a, b, c,
    alpha, beta,
    seq_data,     # dict: a1,b1,c1,a2,b2,c2,a0,b0,c0
    V0, V1, V2,   # fasores RMS (complex)
    frame_step=6,
    traj_stride=3,
    clarke_label="power"
):
    N = len(t)

    # frames reduzidos
    frame_idxs = list(range(0, N, frame_step))
    if frame_idxs[-1] != N - 1:
        frame_idxs.append(N - 1)

    # slider com poucos steps (não polui)
    max_slider_steps = 60
    if len(frame_idxs) > max_slider_steps:
        slider_idxs = np.linspace(0, len(frame_idxs) - 1, max_slider_steps).astype(int)
        slider_frame_idxs = [frame_idxs[i] for i in slider_idxs]
    else:
        slider_frame_idxs = frame_idxs

    # limites comuns
    vmax = float(np.max(np.abs([a, b, c])))
    axis_lim = max(1.0, 2.8 * vmax)

    # resultantes (rastros)
    rxt, ryt = resultant_xy_series(a, b, c)

    rx1, ry1 = resultant_xy_series(seq_data["a1"], seq_data["b1"], seq_data["c1"])
    rx2, ry2 = resultant_xy_series(seq_data["a2"], seq_data["b2"], seq_data["c2"])
    rx0, ry0 = resultant_xy_series(seq_data["a0"], seq_data["b0"], seq_data["c0"])

    # limite do plano αβ
    ab_lim = max(1.0, 1.1 * float(np.max(np.sqrt(alpha**2 + beta**2))))

    # limite do campo de sequências
    seq_lim = max(1.0, 1.2 * float(np.max(np.abs([rx1, ry1, rx2, ry2, rx0, ry0]))))

    # =====================================================
    # FIG: 3x2
    # =====================================================
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            "ABC (tempo)",
            "Campo ABC (XY) — rastro do resultante",
            f"Clarke αβ (tempo) — {clarke_label}",
            "Plano αβ — rastro",
            "Campo das Sequências (XY) — V1, V2, V0 com rastros",
            "|V1|, |V2|, |V0| (RMS)"
        ),
        horizontal_spacing=0.10,
        vertical_spacing=0.12,
    )

    # =====================================================
    # TRACES ESTÁTICOS
    # =====================================================
    # (1,1) ABC tempo
    fig.add_trace(go.Scatter(x=t, y=a, mode="lines", name="A"), row=1, col=1)  # 0
    fig.add_trace(go.Scatter(x=t, y=b, mode="lines", name="B"), row=1, col=1)  # 1
    fig.add_trace(go.Scatter(x=t, y=c, mode="lines", name="C"), row=1, col=1)  # 2

    # (2,1) Clarke tempo
    fig.add_trace(go.Scatter(x=t, y=alpha, mode="lines", name="α"), row=2, col=1)  # 3
    fig.add_trace(go.Scatter(x=t, y=beta,  mode="lines", name="β"), row=2, col=1)  # 4

    # (2,2) Plano αβ: base de trajetória
    fig.add_trace(go.Scatter(x=alpha, y=beta, mode="lines", name="traj αβ", line=dict(width=2)),
                  row=2, col=2)  # 5
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="origem", marker=dict(size=7)),
                  row=2, col=2)  # 6

    # (3,2) RMS bars (visíveis!)
    rms_vals = [float(np.abs(V1)), float(np.abs(V2)), float(np.abs(V0))]
    rms_labels = ["|V1|", "|V2|", "|V0|"]
    fig.add_trace(go.Bar(
        x=rms_labels,
        y=rms_vals,
        name="RMS",
        text=[f"{v:.3f}" for v in rms_vals],
        textposition="outside",
        marker=dict(
            color="rgba(88,166,255,0.85)",
            line=dict(color="rgba(240,246,252,0.35)", width=1)
        ),
    ), row=3, col=2)  # 7

    # =====================================================
    # TRACES DINÂMICOS (inicial)
    # =====================================================
    i0 = frame_idxs[0]

    # markers ABC @t
    fig.add_trace(go.Scatter(x=[t[i0]], y=[a[i0]], mode="markers", name="A@t", marker=dict(size=10)),
                  row=1, col=1)  # 8
    fig.add_trace(go.Scatter(x=[t[i0]], y=[b[i0]], mode="markers", name="B@t", marker=dict(size=10)),
                  row=1, col=1)  # 9
    fig.add_trace(go.Scatter(x=[t[i0]], y=[c[i0]], mode="markers", name="C@t", marker=dict(size=10)),
                  row=1, col=1)  # 10

    # Campo ABC vetores + resultante + rastro
    vecs0, _ = abc_vectors_xy(a[i0], b[i0], c[i0])
    colors_abc = ["#FF5555", "#55FF55", "#5555FF"]
    names_abc = ["A", "B", "C"]

    # vetores A,B,C
    for (vx, vy), colr, nm in zip(vecs0, colors_abc, names_abc):
        fig.add_trace(go.Scatter(
            x=[0, vx], y=[0, vy],
            mode="lines+markers", name=nm,
            line=dict(width=4, color=colr),
            marker=dict(size=7)
        ), row=1, col=2)  # 11,12,13

    # resultante
    fig.add_trace(go.Scatter(
        x=[0, rxt[i0]], y=[0, ryt[i0]],
        mode="lines+markers", name="Resultante",
        line=dict(width=5, color="white"),
        marker=dict(size=9)
    ), row=1, col=2)  # 14

    # rastro resultante
    fig.add_trace(go.Scatter(
        x=rxt[:i0+1:traj_stride], y=ryt[:i0+1:traj_stride],
        mode="lines", name="rastro ABC",
        line=dict(width=2, color="rgba(240,246,252,0.35)")
    ), row=1, col=2)  # 15

    # Clarke markers @t
    fig.add_trace(go.Scatter(x=[t[i0]], y=[alpha[i0]], mode="markers", name="α@t", marker=dict(size=9)),
                  row=2, col=1)  # 16
    fig.add_trace(go.Scatter(x=[t[i0]], y=[beta[i0]],  mode="markers", name="β@t", marker=dict(size=9)),
                  row=2, col=1)  # 17

    # Plano αβ ponto + vetor
    fig.add_trace(go.Scatter(x=[alpha[i0]], y=[beta[i0]], mode="markers", name="ponto αβ", marker=dict(size=11)),
                  row=2, col=2)  # 18
    fig.add_trace(go.Scatter(x=[0, alpha[i0]], y=[0, beta[i0]], mode="lines", name="vetor αβ",
                             line=dict(width=3, dash="dot", color="rgba(255,215,0,0.9)")),
                  row=2, col=2)  # 19

    # Campo sequências (MESMO gráfico): V1, V2, V0 (vetor + rastro cada)
    # V1
    fig.add_trace(go.Scatter(
        x=[0, rx1[i0]], y=[0, ry1[i0]],
        mode="lines+markers", name="V1 (vetor)",
        line=dict(width=4, color="rgba(88,166,255,0.95)"),
        marker=dict(size=8)
    ), row=3, col=1)  # 20
    fig.add_trace(go.Scatter(
        x=rx1[:i0+1:traj_stride], y=ry1[:i0+1:traj_stride],
        mode="lines", name="rastro V1",
        line=dict(width=2, color="rgba(88,166,255,0.35)")
    ), row=3, col=1)  # 21

    # V2
    fig.add_trace(go.Scatter(
        x=[0, rx2[i0]], y=[0, ry2[i0]],
        mode="lines+markers", name="V2 (vetor)",
        line=dict(width=4, color="rgba(170,170,170,0.95)"),
        marker=dict(size=8)
    ), row=3, col=1)  # 22
    fig.add_trace(go.Scatter(
        x=rx2[:i0+1:traj_stride], y=ry2[:i0+1:traj_stride],
        mode="lines", name="rastro V2",
        line=dict(width=2, color="rgba(170,170,170,0.35)")
    ), row=3, col=1)  # 23

    # V0
    fig.add_trace(go.Scatter(
        x=[0, rx0[i0]], y=[0, ry0[i0]],
        mode="lines+markers", name="V0 (vetor)",
        line=dict(width=4, color="rgba(255,215,0,0.95)"),
        marker=dict(size=8)
    ), row=3, col=1)  # 24
    fig.add_trace(go.Scatter(
        x=rx0[:i0+1:traj_stride], y=ry0[:i0+1:traj_stride],
        mode="lines", name="rastro V0",
        line=dict(width=2, color="rgba(255,215,0,0.35)")
    ), row=3, col=1)  # 25

    # índices dinâmicos (8..25)
    dynamic_trace_idxs = list(range(8, 26))

    # =====================================================
    # FRAMES
    # =====================================================
    frames = []
    for k in frame_idxs:
        vecs, _ = abc_vectors_xy(a[k], b[k], c[k])

        tx  = rxt[:k+1:traj_stride];  ty  = ryt[:k+1:traj_stride]
        t1x = rx1[:k+1:traj_stride];  t1y = ry1[:k+1:traj_stride]
        t2x = rx2[:k+1:traj_stride];  t2y = ry2[:k+1:traj_stride]
        t0x = rx0[:k+1:traj_stride];  t0y = ry0[:k+1:traj_stride]

        frame_data = []

        # 8..10 markers ABC @t
        frame_data.append(go.Scatter(x=[t[k]], y=[a[k]]))
        frame_data.append(go.Scatter(x=[t[k]], y=[b[k]]))
        frame_data.append(go.Scatter(x=[t[k]], y=[c[k]]))

        # 11..13 vetores A,B,C
        for (vx, vy) in vecs:
            frame_data.append(go.Scatter(x=[0, vx], y=[0, vy]))

        # 14 resultante
        frame_data.append(go.Scatter(x=[0, rxt[k]], y=[0, ryt[k]]))
        # 15 rastro ABC
        frame_data.append(go.Scatter(x=tx, y=ty))

        # 16..17 markers Clarke
        frame_data.append(go.Scatter(x=[t[k]], y=[alpha[k]]))
        frame_data.append(go.Scatter(x=[t[k]], y=[beta[k]]))

        # 18 ponto αβ
        frame_data.append(go.Scatter(x=[alpha[k]], y=[beta[k]]))
        # 19 vetor αβ
        frame_data.append(go.Scatter(x=[0, alpha[k]], y=[0, beta[k]]))

        # 20 V1 vetor
        frame_data.append(go.Scatter(x=[0, rx1[k]], y=[0, ry1[k]]))
        # 21 rastro V1
        frame_data.append(go.Scatter(x=t1x, y=t1y))

        # 22 V2 vetor
        frame_data.append(go.Scatter(x=[0, rx2[k]], y=[0, ry2[k]]))
        # 23 rastro V2
        frame_data.append(go.Scatter(x=t2x, y=t2y))

        # 24 V0 vetor
        frame_data.append(go.Scatter(x=[0, rx0[k]], y=[0, ry0[k]]))
        # 25 rastro V0
        frame_data.append(go.Scatter(x=t0x, y=t0y))

        frames.append(go.Frame(data=frame_data, name=str(k), traces=dynamic_trace_idxs))

    fig.frames = frames

    # =====================================================
    # SLIDER (sem labels -> sem sobreposição!)
    # =====================================================
    slider_steps = []
    for k in slider_frame_idxs:
        slider_steps.append(dict(
            method="animate",
            args=[[str(k)], {
                "mode": "immediate",
                "frame": {"duration": 0, "redraw": False},
                "transition": {"duration": 0}
            }],
            label=""
        ))

    # =====================================================
    # LAYOUT / ESTILO
    # =====================================================
    fig.update_layout(
        height=1100,
        margin=dict(l=20, r=20, t=90, b=150),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=FONT_CLR, size=13),

        legend=dict(
            orientation="h",
            y=1.02, x=1,
            xanchor="right", yanchor="bottom",
            font=dict(color=LEG_CLR)
        ),

        updatemenus=[dict(
            type="buttons",
            direction="left",
            x=0.0, y=1.12,
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

        sliders=[dict(
            x=0.0, y=-0.08, len=1.0,
            pad=dict(t=12, b=0),
            currentvalue=dict(prefix="t = ", suffix=" s", font=dict(size=14, color=FONT_CLR), visible=True),
            bgcolor=SL_BG,
            bordercolor=SL_BRD,
            borderwidth=1,
            steps=slider_steps
        )],
    )

    # grid/zeroline
    fig.update_xaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_CLR, zeroline=True, zerolinecolor=ZERO_CLR)

    # Eixos / limites
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

    fig.update_yaxes(title_text="RMS", row=3, col=2)

    # Ajuste para BAR não sumir
    ymax = max(rms_vals) * 1.35
    if ymax < 1e-6:
        ymax = 1.0
    fig.update_yaxes(range=[0, ymax], row=3, col=2)

    # =====================================================
    # CAPTIONS (descrição abaixo de cada gráfico)
    # =====================================================
    def add_caption(xref, yref, text):
        fig.add_annotation(
            x=0.5, y=-0.20,
            xref=xref, yref=yref,
            text=text,
            showarrow=False,
            font=dict(size=12, color="rgba(230,237,243,0.75)"),
            align="center"
        )

    # x1/y1, x2/y2, x3/y3, x4/y4, x5/y5, x6/y6
    add_caption("x1 domain", "y1 domain", "ABC (tempo): formas de onda trifásicas (A, B, C) no domínio do tempo.")
    add_caption("x2 domain", "y2 domain", "Campo ABC (XY): vetores A/B/C e vetor resultante, com rastro do resultante.")
    add_caption("x3 domain", "y3 domain", "Clarke αβ (tempo): componentes α e β no domínio do tempo.")
    add_caption("x4 domain", "y4 domain", "Plano αβ: trajetória αβ e vetor instantâneo (origem → ponto atual).")
    add_caption("x5 domain", "y5 domain", "Campo das sequências: V1 (positiva), V2 (negativa) e V0 (zero) no mesmo plano, com rastros.")
    add_caption("x6 domain", "y6 domain", "RMS: magnitudes |V1|, |V2|, |V0| estimadas na fundamental (fasores).")

    return fig


# =========================================================
# UI
# =========================================================
st.markdown("## ⚡ IEEE 34 Barras — V7 (Campos + Rastros + Sequências no MESMO gráfico)")
st.caption("Upload múltiplo • Play/Pause dark • Rastros • V0/V1/V2 juntos • Captions abaixo dos gráficos • FFT/THD opcional")


# =========================================================
# SIDEBAR
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
    selected_file = st.selectbox("Arquivo ativo", file_names, format_func=lambda x: f"📄 {x}")
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
    st.markdown("### 🔁 Componentes Simétricas (V0/V1/V2)")
    seq_f0 = st.number_input("f0 p/ fasor (Hz)", value=60.0, min_value=1.0, step=1.0)
    seq_window = st.selectbox("Janela p/ fasor", ["hann", "rect"], index=0)

    st.divider()
    st.markdown("### 🎞️ Animação / Performance")
    frame_step = st.slider("frame_step (↓ mais suave)", 1, 60, 6, 1)
    traj_stride = st.slider("traj_stride (↓ rastro mais denso)", 1, 20, 3, 1)
    st.caption("Se travar: aumente frame_step e traj_stride.")

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
    st.error(f"Não encontrei `ts_{point}.Time/Data` dentro de `ts` no arquivo `{selected_file}`.")
    st.stop()

t = np.asarray(t).squeeze()

# Exigimos N×3
if not (isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] >= 3):
    st.error(f"Esta versão exige sinal trifásico N×3. Recebi x.shape={getattr(x, 'shape', None)}.")
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
# SEQUÊNCIAS (fasores RMS + reconstrução fundamental)
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

seq_data = {
    "a0": a0, "b0": b0, "c0": c0,
    "a1": a1, "b1": b1, "c1": c1,
    "a2": a2, "b2": b2, "c2": c2
}


# =========================================================
# MAIN VIEW
# =========================================================
st.markdown(
    f"### 🧪 Caso selecionado\n"
    f"- **Arquivo:** `{selected_file}`\n"
    f"- **Ponto:** `{point}`\n"
    f"- **m1:** `{m1}`\n"
    f"- **fs estimada:** `{fs:.2f} Hz`\n"
    f"- **N:** `{len(t)}` | **frame_step:** `{frame_step}` | **traj_stride:** `{traj_stride}`"
)

st.markdown("#### 📌 Componentes Simétricas (RMS) na fundamental")
st.table({
    "Componente": ["V1 (Positiva)", "V2 (Negativa)", "V0 (Zero)"],
    "|V| RMS": [float(np.abs(V1)), float(np.abs(V2)), float(np.abs(V0))],
    "∠ (deg)": [float(np.angle(V1, deg=True)), float(np.angle(V2, deg=True)), float(np.angle(V0, deg=True))]
})

fig = build_animated_figure(
    t=t, a=a, b=b, c=c,
    alpha=alpha, beta=beta,
    seq_data=seq_data,
    V0=V0, V1=V1, V2=V2,
    frame_step=frame_step,
    traj_stride=traj_stride,
    clarke_label=("power" if clarke_mode_key == "power" else "amp")
)

st.plotly_chart(fig, use_container_width=True)


# =========================================================
# FFT / THD (optional)
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
