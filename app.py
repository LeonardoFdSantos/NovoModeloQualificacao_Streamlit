import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.signal import windows
from scipy.fft import fft, fftfreq


# -----------------------------
# Config Streamlit
# -----------------------------
st.set_page_config(
    page_title="IEEE34 – Visualizador .MAT (ABC/Clarke/FFT/THD)",
    layout="wide"
)

# -----------------------------
# Funções Núcleo: Clarke, FFT, THD
# -----------------------------
def clarke_transform(a, b, c, mode="power"):
    """
    Transformada de Clarke (alpha-beta).
    mode:
      - "power": k = sqrt(2/3)  (invariante de potência)
      - "amp":   k = 2/3        (invariante de amplitude)
    """
    if mode == "amp":
        k = 2/3
    else:
        k = np.sqrt(2/3)

    alpha = k * (a - 0.5*b - 0.5*c)
    beta  = k * ((np.sqrt(3)/2)*b - (np.sqrt(3)/2)*c)
    return alpha, beta


def compute_fft_rms(signal, fs, window="hann", remove_mean=True):
    """
    FFT unilateral com magnitude RMS.
    - remove_mean: remove componente DC antes
    - window: 'hann' ou 'rect'
    Retorna: freqs (>=0), X_rms
    """
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

    # Correção de ganho da janela
    # (2/sum(w)) -> unilateral (exceto DC/Nyquist; aqui usamos forma prática)
    X_mag = (2.0 / np.sum(w)) * np.abs(X)
    X_rms = X_mag / np.sqrt(2)

    return freqs, X_rms


def compute_thd_percent(freqs, spectrum_rms, f_fund=60.0, h_max=25, tol_hz=1.0):
    """
    THD% = sqrt(sum_{h=2..H} Xh^2) / X1 * 100
    - freqs: eixo de frequência
    - spectrum_rms: magnitude RMS
    - tol_hz: tolerância para achar bin próximo da harmônica
    """
    if len(freqs) == 0:
        return np.nan

    freqs = np.asarray(freqs)
    spec = np.asarray(spectrum_rms)

    # Fundamental
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

    thd = np.sqrt(harm_sq) / X1
    return float(thd * 100.0)


# -----------------------------
# Parser do .MAT no seu formato
# -----------------------------
def _mat_struct_getfield(obj, name):
    """
    Acessa field em MATLAB struct carregado pelo scipy.
    Lida com:
      - mat_struct (atributo)
      - ndarray dtype.names
    """
    # caso 1: mat_struct com atributo
    if hasattr(obj, name):
        return getattr(obj, name)

    # caso 2: ndarray/record com dtype.names
    try:
        if hasattr(obj, "dtype") and obj.dtype.names and name in obj.dtype.names:
            return obj[name]
    except Exception:
        pass

    return None


def extract_ts_from_mat(mat, point_name):
    """
    Extrai Time/Data do formato:
      mat['ts'] -> struct
      mat['ts'].ts_I_800.Time / .Data
    Retorna: (t, X) ou (None, None)

    X pode ser:
      - (N,3) trifásico
      - (N,) monofásico
    """
    if 'ts' not in mat:
        return None, None

    ts_root = mat['ts']

    # SciPy pode trazer 'ts' como:
    # - mat_struct
    # - ndarray object (1,1)
    # vamos tentar normalizar
    if isinstance(ts_root, np.ndarray):
        # muitas vezes vem (1,1)
        try:
            ts_root = ts_root.squeeze()
            # se virou array de objetos, pega o item
            if isinstance(ts_root, np.ndarray) and ts_root.dtype == object:
                ts_root = ts_root.item()
        except Exception:
            pass

    key = f"ts_{point_name}"
    entry = _mat_struct_getfield(ts_root, key)
    if entry is None:
        return None, None

    # entry pode vir como ndarray (1,1) ou mat_struct direto
    if isinstance(entry, np.ndarray):
        entry = entry.squeeze()
        if isinstance(entry, np.ndarray) and entry.dtype == object:
            entry = entry.item()

    time = _mat_struct_getfield(entry, "Time")
    data = _mat_struct_getfield(entry, "Data")
    if time is None or data is None:
        return None, None

    t = np.array(time).squeeze()
    x = np.array(data)
    x = np.squeeze(x)

    # Força consistência:
    # - se trifásico, esperamos (N,3)
    # - se vier (3,N), transpor
    if x.ndim == 2 and x.shape[0] == 3 and x.shape[1] == t.shape[0]:
        x = x.T

    return t, x


def extract_m1_location(mat):
    if 'm1_location' not in mat:
        return None
    try:
        return float(np.array(mat['m1_location']).squeeze())
    except Exception:
        return None


# -----------------------------
# UI
# -----------------------------
st.title("📈 IEEE34 – Visualizador de Casos (.mat) com Clarke, FFT e THD")

with st.sidebar:
    st.header("Upload e Seleção")

    uploaded = st.file_uploader(
        "Envie um ou vários arquivos .mat",
        type=["mat"],
        accept_multiple_files=True
    )

    points = ['I_800','V_800','I_T2F','V_T2F','I_818','V_818','I_820','V_820','I_822','V_822']
    point = st.selectbox("Ponto", points, index=0)

    st.divider()
    st.header("Processamento")

    remove_mean = st.checkbox("Remover offset (subtrair média)", value=True)

    clarke_mode = st.radio(
        "Clarke (escala k)",
        ["power (√2/3)", "amp (2/3)"],
        index=0
    )
    clarke_mode_key = "power" if clarke_mode.startswith("power") else "amp"

    st.divider()
    st.header("FFT / THD")

    show_fft = st.checkbox("Mostrar FFT", value=True)
    window_type = st.selectbox("Janela", ["hann", "rect"], index=0)

    f_fund = st.number_input("Frequência fundamental (Hz)", value=60.0, min_value=1.0, step=1.0)
    h_max = st.slider("Máx. harmônica (THD)", min_value=5, max_value=60, value=25, step=1)
    tol_hz = st.number_input("Tolerância p/ achar harmônica (Hz)", value=1.0, min_value=0.1, step=0.1)

    fft_xmax = st.number_input("Limite X da FFT (Hz)", value=float(h_max)*float(f_fund) + 10.0, min_value=10.0, step=10.0)


if not uploaded:
    st.info("Envie arquivos .mat na barra lateral para começar.")
    st.stop()

# Carrega tudo em memória (por upload)
# Map: filename -> mat dict
mats = {}
for uf in uploaded:
    try:
        mats[uf.name] = loadmat(io.BytesIO(uf.getvalue()), squeeze_me=False, struct_as_record=False)
    except Exception as e:
        st.error(f"Falha ao ler {uf.name}: {e}")

if not mats:
    st.error("Nenhum arquivo válido carregado.")
    st.stop()

# Seleção do arquivo
file_names = sorted(mats.keys())
selected_file = st.sidebar.selectbox("Arquivo", file_names, index=0)
mat = mats[selected_file]

# Meta
m1 = extract_m1_location(mat)

# Layout principal
col_left, col_right = st.columns([1.2, 1.0], gap="large")

with col_left:
    st.subheader("Sinais no tempo")

    t, x = extract_ts_from_mat(mat, point)
    if t is None or x is None or len(np.atleast_1d(t)) == 0:
        st.warning(f"Não encontrei ts_{point}.Time/Data dentro de {selected_file}.")
        st.stop()

    t = np.asarray(t).squeeze()

    # Preparar A/B/C
    if x.ndim == 1:
        a = np.asarray(x).squeeze()
        b = np.zeros_like(a)
        c = np.zeros_like(a)
        is_three_phase = False
    elif x.ndim == 2 and x.shape[1] >= 3:
        a = x[:, 0]
        b = x[:, 1]
        c = x[:, 2]
        is_three_phase = True
    else:
        # fallback
        a = np.asarray(x).squeeze()
        b = np.zeros_like(a)
        c = np.zeros_like(a)
        is_three_phase = False

    if remove_mean:
        a = a - np.mean(a)
        b = b - np.mean(b)
        c = c - np.mean(c)

    # fs
    dt = np.diff(t)
    fs = 1.0 / float(np.mean(dt)) if len(dt) > 0 else 0.0

    # Plot ABC
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(t, a, label="A")
    ax1.plot(t, b, label="B")
    ax1.plot(t, c, label="C")
    ax1.set_title(f"ABC – {point} (fs≈{fs:.2f} Hz) | m1={m1}")
    ax1.set_xlabel("Tempo (s)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    st.pyplot(fig1)

    # Clarke (se trifásico real)
    if is_three_phase:
        alpha, beta = clarke_transform(a, b, c, mode=clarke_mode_key)

        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(t, alpha, label="α")
        ax2.plot(t, beta, label="β")
        ax2.set_title(f"Clarke αβ – {point} ({clarke_mode})")
        ax2.set_xlabel("Tempo (s)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("Sinal não parece trifásico (N×3). Clarke αβ foi omitido.")


with col_right:
    st.subheader("Metadados e Análise Espectral")

    st.markdown(
        f"""
        **Arquivo:** `{selected_file}`  
        **Ponto:** `{point}`  
        **m1_location:** `{m1}`  
        **fs estimado:** `{fs:.3f} Hz`  
        """
    )

    if show_fft:
        if fs <= 0 or len(a) < 8:
            st.warning("Não foi possível calcular FFT (fs inválido ou sinal muito curto).")
        else:
            freqa, F_a = compute_fft_rms(a, fs, window=window_type, remove_mean=False)
            freqb, F_b = compute_fft_rms(b, fs, window=window_type, remove_mean=False)
            freqc, F_c = compute_fft_rms(c, fs, window=window_type, remove_mean=False)

            # THD
            thd_a = compute_thd_percent(freqa, F_a, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)
            thd_b = compute_thd_percent(freqb, F_b, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)
            thd_c = compute_thd_percent(freqc, F_c, f_fund=f_fund, h_max=h_max, tol_hz=tol_hz)

            st.markdown("### THD (%)")
            st.table({
                "Fase": ["A", "B", "C"],
                "THD (%)": [
                    None if np.isnan(thd_a) else round(thd_a, 2),
                    None if np.isnan(thd_b) else round(thd_b, 2),
                    None if np.isnan(thd_c) else round(thd_c, 2),
                ]
            })

            # Plot FFT
            st.markdown("### FFT (Magnitude RMS)")
            fig3, ax3 = plt.subplots(figsize=(8.5, 4.2))
            ax3.plot(freqa, F_a, label="A")
            ax3.plot(freqb, F_b, label="B")
            ax3.plot(freqc, F_c, label="C")
            ax3.set_xlim(0, float(fft_xmax))
            ax3.set_title(f"FFT RMS – janela: {window_type}")
            ax3.set_xlabel("Frequência (Hz)")
            ax3.set_ylabel("Amplitude (RMS)")
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            st.pyplot(fig3)

            # Dica: mostrar top harmônicas
            st.markdown("### Top componentes (aprox.)")
            def top_peaks(freqs, spec, n=8, fmax=None):
                if len(freqs) == 0:
                    return []
                mask = freqs <= (fmax if fmax is not None else freqs[-1])
                f2 = freqs[mask]
                s2 = spec[mask]
                if len(s2) == 0:
                    return []
                idxs = np.argsort(s2)[::-1][:n]
                out = [(float(f2[i]), float(s2[i])) for i in idxs]
                return out

            peaksA = top_peaks(freqa, F_a, n=8, fmax=float(fft_xmax))
            st.write("**A:**", peaksA[:6])
    else:
        st.info("FFT desativada na barra lateral.")


st.caption("Dica: se algum arquivo não carregar, é quase sempre diferença de estrutura do MATLAB (struct/logging). Eu adapto o parser se você me mandar um exemplo do .mat problemático.")
