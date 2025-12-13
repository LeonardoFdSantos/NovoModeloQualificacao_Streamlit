import io
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from scipy.io import loadmat
from scipy.signal import windows
from scipy.fft import fft, fftfreq


# -----------------------------
# Streamlit config
# -----------------------------
st.set_page_config(
    page_title="IEEE34 – Visualizador .MAT (Plotly/Clarke/FFT/THD)",
    layout="wide"
)


# -----------------------------
# Core math
# -----------------------------
def clarke_transform(a, b, c, mode="power"):
    if mode == "amp":
        k = 2/3
    else:
        k = np.sqrt(2/3)
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

    if window == "hann":
        w = windows.hann(N)
    else:
        w = np.ones(N)

    xw = x * w
    X = fft(xw)
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


# -----------------------------
# Robust-ish MATLAB struct access
# -----------------------------
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
        try:
            ts_root = ts_root.squeeze()
            if isinstance(ts_root, np.ndarray) and ts_root.dtype == object:
                ts_root = ts_root.item()
        except Exception:
            pass

    key = f"ts_{point_name}"
    entry = _mat_getfield(ts_root, key)
    if entry is None:
        return None, None

    if isinstance(entry, np.ndarray):
        entry = entry.squeeze()
        if isinstance(entry, np.ndarray) and entry.dtype == object:
            entry = entry.item()

    time = _mat_getfield(entry, "Time")
    data = _mat_getfield(entry, "Data")
    if time is None or data is None:
        return None, None

    t = np.asarray(time).squeeze()
    x = np.asarray(data)
    x = np.squeeze(x)

    # se vier (3,N) e t for (N,), transpõe
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


# -----------------------------
# Plotly helpers
# -----------------------------
def fig_timeseries_abc(t, a, b, c, idx=None, title="ABC"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=a, mode="lines", name="A"))
    fig.add_trace(go.Scatter(x=t, y=b, mode="lines", name="B"))
    fig.add_trace(go.Scatter(x=t, y=c, mode="lines", name="C"))

    if idx is not None:
        fig.add_trace(go.Scatter(x=[t[idx]], y=[a[idx]], mode="markers", name="A @t", marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[t[idx]], y=[b[idx]], mode="markers", name="B @t", marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[t[idx]], y=[c[idx]], mode="markers", name="C @t", marker=dict(size=10)))

    fig.update_layout(
        title=title,
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        height=360,
        margin=dict(l=30, r=20, t=50, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)")
    return fig


def fig_timeseries_ab(t, alpha, beta, idx=None, title="Clarke αβ (tempo)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=alpha, mode="lines", name="α"))
    fig.add_trace(go.Scatter(x=t, y=beta, mode="lines", name="β"))
    if idx is not None:
        fig.add_trace(go.Scatter(x=[t[idx]], y=[alpha[idx]], mode="markers", name="α @t", marker=dict(size=10)))
        fig.add_trace(go.Scatter(x=[t[idx]], y=[beta[idx]], mode="markers", name="β @t", marker=dict(size=10)))
    fig.update_layout(
        title=title,
        xaxis_title="Tempo (s)",
        yaxis_title="Amplitude",
        height=320,
        margin=dict(l=30, r=20, t=50, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)")
    return fig


def fig_alpha_beta_plane(alpha, beta, idx=None, show_traj=True, title="Plano αβ"):
    fig = go.Figure()

    if show_traj:
        fig.add_trace(go.Scatter(
            x=alpha, y=beta, mode="lines", name="traj",
            line=dict(width=2)
        ))

    if idx is not None:
        fig.add_trace(go.Scatter(
            x=[alpha[idx]], y=[beta[idx]],
            mode="markers", name="ponto atual",
            marker=dict(size=12)
        ))

        # vetor (0,0) -> (alpha,beta)
        fig.add_trace(go.Scatter(
            x=[0, alpha[idx]], y=[0, beta[idx]],
            mode="lines", name="vetor",
            line=dict(width=3, dash="dot")
        ))

    # eixos
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="origem", marker=dict(size=10)))

    fig.update_layout(
        title=title,
        xaxis_title="α",
        yaxis_title="β",
        height=420,
        margin=dict(l=30, r=20, t=50, b=35),
        showlegend=True
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)", zeroline=True, zerolinecolor="rgba(220,220,220,0.25)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)", zeroline=True, zerolinecolor="rgba(220,220,220,0.25)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def fig_fft(freqs, Fa, Fb, Fc, xmax, title="FFT RMS"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=freqs, y=Fa, mode="lines", name="A"))
    fig.add_trace(go.Scatter(x=freqs, y=Fb, mode="lines", name="B"))
    fig.add_trace(go.Scatter(x=freqs, y=Fc, mode="lines", name="C"))
    fig.update_layout(
        title=title,
        xaxis_title="Frequência (Hz)",
        yaxis_title="Amplitude (RMS)",
        height=360,
        margin=dict(l=30, r=20, t=50, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[0, float(xmax)], showgrid=True, gridcolor="rgba(200,200,200,0.15)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.15)")
    return fig


# -----------------------------
# UI
# -----------------------------
st.title("⚡ Visualizador IEEE34 (.mat) — Plotly + Clarke + FFT + THD")

with st.sidebar:
    st.header("Upload")
    uploaded = st.file_uploader(
        "Envie um ou vários arquivos .mat",
        type=["mat"],
        accept_multiple_files=True
    )

    points = ['I_800','V_800','I_T2F','V_T2F','I_818','V_818','I_820','V_820','I_822','V_822']
    point = st.selectbox("Ponto", points, index=0)

    st.divider()
    st.header("Interatividade")
    remove_mean = st.checkbox("Remover offset", value=True)
    clarke_mode = st.radio("Clarke", ["power (√2/3)", "amp (2/3)"], index=0)
    clarke_mode_key = "power" if clarke_mode.startswith("power") else "amp"
    show_traj = st.checkbox("Mostrar trajetória αβ", value=True)

    st.divider()
    st.header("FFT/THD")
    show_fft = st.checkbox("Mostrar FFT", value=True)
    window_type = st.selectbox("Janela", ["hann", "rect"], index=0)
    f_fund = st.number_input("Fundamental (Hz)", value=60.0, min_value=1.0, step=1.0)
    h_max = st.slider("Máx. harmônica THD", 5, 60, 25, 1)
    tol_hz = st.number_input("Tolerância harmônica (Hz)", value=1.0, min_value=0.1, step=0.1)
    fft_xmax = st.number_input("Limite X FFT (Hz)", value=float(h_max)*float(f_fund) + 10.0, min_value=10.0, step=10.0)

if not uploaded:
    st.info("Envie arquivos .mat no painel lateral para começar.")
    st.stop()

# Carregar mats
mats = {}
bad = []
for uf in uploaded:
    try:
        mats[uf.name] = loadmat(io.BytesIO(uf.getvalue()), squeeze_me=False, struct_as_record=False)
    except Exception as e:
        bad.append((uf.name, str(e)))

if bad:
    st.warning("Alguns arquivos falharam ao carregar:")
    for name, err in bad:
        st.write(f"- {name}: {err}")

if not mats:
    st.error("Nenhum arquivo válido foi carregado.")
    st.stop()

file_names = sorted(mats.keys())
selected_file = st.selectbox("Escolha o arquivo", file_names, index=0)
mat = mats[selected_file]

m1 = extract_m1_location(mat)

t, x = extract_ts_from_mat(mat, point)
if t is None or x is None or len(np.atleast_1d(t)) == 0:
    st.error(f"Não encontrei ts_{point}.Time/Data em {selected_file}.")
    st.stop()

t = np.asarray(t).squeeze()

# Preparar A/B/C
if x.ndim == 1:
    a = np.asarray(x).squeeze()
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    is_three_phase = False
elif x.ndim == 2 and x.shape[1] >= 3:
    a, b, c = x[:, 0], x[:, 1], x[:, 2]
    is_three_phase = True
else:
    a = np.asarray(x).squeeze()
    b = np.zeros_like(a)
    c = np.zeros_like(a)
    is_three_phase = False

if remove_mean:
    a = a - np.mean(a)
    b = b - np.mean(b)
    c = c - np.mean(c)

dt = np.diff(t)
fs = 1.0 / float(np.mean(dt)) if len(dt) > 0 else 0.0

# Slider de tempo (frame)
idx = st.slider("Tempo (frame)", 0, len(t)-1, 0, 1, format="%d")
st.caption(f"Arquivo: `{selected_file}` | Ponto: `{point}` | m1={m1} | fs≈{fs:.2f} Hz | t={t[idx]:.6f}s")

# Clarke
if is_three_phase:
    alpha, beta = clarke_transform(a, b, c, mode=clarke_mode_key)
else:
    alpha = beta = None

# Layout de gráficos
c1, c2 = st.columns([1.3, 1.0], gap="large")

with c1:
    st.plotly_chart(
        fig_timeseries_abc(t, a, b, c, idx=idx, title=f"ABC (tempo) — {point}"),
        use_container_width=True
    )

    if is_three_phase:
        st.plotly_chart(
            fig_timeseries_ab(t, alpha, beta, idx=idx, title=f"Clarke αβ (tempo) — {clarke_mode}"),
            use_container_width=True
        )
    else:
        st.info("Sinal não trifásico (N×3). Clarke foi omitido.")

with c2:
    if is_three_phase:
        st.plotly_chart(
            fig_alpha_beta_plane(alpha, beta, idx=idx, show_traj=show_traj, title="Plano αβ (trajetória + vetor)"),
            use_container_width=True
        )

    if show_fft and fs > 0 and len(a) >= 8:
        freqs, Fa = compute_fft_rms(a, fs, window=window_type, remove_mean=False)
        _, Fb = compute_fft_rms(b, fs, window=window_type, remove_mean=False)
        _, Fc = compute_fft_rms(c, fs, window=window_type, remove_mean=False)

        thd_a = compute_thd_percent(freqs, Fa, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)
        thd_b = compute_thd_percent(freqs, Fb, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)
        thd_c = compute_thd_percent(freqs, Fc, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)

        st.subheader("THD (%)")
        st.table({
            "Fase": ["A", "B", "C"],
            "THD (%)": [
                None if np.isnan(thd_a) else round(thd_a, 2),
                None if np.isnan(thd_b) else round(thd_b, 2),
                None if np.isnan(thd_c) else round(thd_c, 2),
            ]
        })

        st.plotly_chart(
            fig_fft(freqs, Fa, Fb, Fc, xmax=fft_xmax, title=f"FFT RMS — janela {window_type}"),
            use_container_width=True
        )
    else:
        st.info("FFT desativada ou fs inválido/sinal curto.")
