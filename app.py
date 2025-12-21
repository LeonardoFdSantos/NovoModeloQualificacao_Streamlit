import streamlit as st
import numpy as np
import scipy.io as sio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# =========================================================
# 📚 1. CONFIGURAÇÕES
# =========================================================
st.set_page_config(page_title="T2F Master Suite", layout="wide", page_icon="⚡")

CURVES = {
    "IEC Standard Inverse":  (0.14, 0.0, 0.02),
    "IEC Very Inverse":      (13.5, 0.0, 1.0),
    "IEC Extremely Inverse": (80.0, 0.0, 2.0),
    "IEC Long Time Inverse": (120.0, 0.0, 1.0),
    "IEEE Moderately Inv":   (0.0515, 0.114, 0.02),
    "IEEE Very Inverse":     (19.61, 0.491, 2.0),
    "IEEE Extremely Inv":    (28.2, 0.1217, 2.0)
}

THEME = {
    "A": "#00ffff", "B": "#ff3333", "C": "#00ff00",
    "V1": "#4facfe", "V2": "#f093fb", "V0": "#fcc203",
    "Ref": "#ff6600",
    "bg": "#1e1e1e", "grid": "#444"
}

# =========================================================
# 🧠 2. FUNÇÕES DE CÁLCULO E CACHE
# =========================================================
@st.cache_data
def get_cached_tcc_curve(pickup, dial, curve_name):
    i_plot = np.logspace(np.log10(0.1), np.log10(30000), 500)
    if curve_name not in CURVES: A, B, p = CURVES["IEC Standard Inverse"]
    else: A, B, p = CURVES[curve_name]
    
    safe_Ip = pickup if pickup > 0 else 0.001
    M = i_plot / safe_Ip
    t_plot = np.full_like(i_plot, 2000.0, dtype=float)
    mask = M > 1.001
    if np.any(mask):
        denom = np.power(M[mask], p) - 1
        denom[denom == 0] = 1e-9
        t_plot[mask] = dial * ( (A / denom) + B )
    return i_plot, t_plot

def calculate_tcc_single(I_val, Ip, TD, curve_name):
    if I_val is None: return 2000.0
    if curve_name not in CURVES: A, B, p = CURVES["IEC Standard Inverse"]
    else: A, B, p = CURVES[curve_name]
    M = I_val / (Ip if Ip > 0 else 0.001)
    if M <= 1.001: return 2000.0
    val = TD * ((A / ((M**p)-1)) + B)
    return min(val, 1000.0)

def parse_mat_file(mat_data):
    parsed = {}
    t = mat_data.get('t') if 't' in mat_data else mat_data.get('time')
    if t is None: return None
    parsed['t'] = t.flatten()
    for key in mat_data.keys():
        if key.startswith('__') or key in ['t', 'time', 'm1']: continue
        base = key
        tipo = 'raw'
        if key.endswith('_rms'): base = key[:-4]; tipo = 'rms'
        elif key.endswith('_phasor'): base = key[:-7]; tipo = 'phasor'
        elif key.endswith('_seq'): base = key[:-4]; tipo = 'seq'
        elif key.endswith('_clarke'): base = key[:-7]; tipo = 'clarke'
        elif key.endswith('_raw'): base = key[:-4]; tipo = 'raw'
        if base not in parsed: parsed[base] = {}
        parsed[base][tipo] = mat_data[key]
    return parsed

@st.cache_data
def get_downsampled_data(t, signal, target_points=1500):
    if signal is None: return t, signal
    n = len(t)
    if n <= target_points: return t, signal
    step = int(n / target_points)
    return t[::step], signal[::step]

# =========================================================
# 📊 3. PLOTS ESTÁTICOS (PARA O PLAYER PYTHON - ABAS 1 e 3)
# =========================================================
def create_waveform_fig(t, sig_rms, title, y_label, time_mark, ref_sig=None, ref_t=None):
    fig = go.Figure()
    t_opt, sig_opt = get_downsampled_data(t, sig_rms)
    
    fig.add_trace(go.Scatter(x=t_opt, y=sig_opt[:,0], name='A', line=dict(color=THEME['A'], width=1.5)))
    fig.add_trace(go.Scatter(x=t_opt, y=sig_opt[:,1], name='B', line=dict(color=THEME['B'], width=1.5)))
    fig.add_trace(go.Scatter(x=t_opt, y=sig_opt[:,2], name='C', line=dict(color=THEME['C'], width=1.5)))
    
    if ref_sig is not None and ref_t is not None:
        tr_opt, sigr_opt = get_downsampled_data(ref_t, ref_sig)
        fig.add_trace(go.Scatter(x=tr_opt, y=sigr_opt[:,0], name='Ref A', line=dict(color=THEME['A'], dash='dash', width=1)))
        fig.add_trace(go.Scatter(x=tr_opt, y=sigr_opt[:,1], name='Ref B', line=dict(color=THEME['B'], dash='dash', width=1)))
        fig.add_trace(go.Scatter(x=tr_opt, y=sigr_opt[:,2], name='Ref C', line=dict(color=THEME['C'], dash='dash', width=1)))

    fig.add_vline(x=time_mark, line_width=2, line_color="white")
    fig.update_layout(title=title, yaxis_title=y_label, height=250, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark")
    return fig

def create_phasor_fig(phasors, rms_vals, title, ref_phasors=None, ref_rms=None):
    fig = go.Figure()
    def add_arrows(ph_vals, mag_vals, suffix="", style_dash=None):
        cols = [THEME['A'], THEME['B'], THEME['C']]
        names = ['A', 'B', 'C']
        for k in range(3):
            ang_rad = np.angle(ph_vals[k])
            r = mag_vals[k]
            x_end = r * np.cos(ang_rad); y_end = r * np.sin(ang_rad)
            fig.add_trace(go.Scatter(x=[0, x_end], y=[0, y_end], mode='lines+markers',
                marker=dict(size=[0, 8], symbol='arrow-bar-up', angle=0),
                line=dict(color=cols[k], width=3, dash=style_dash), name=f"{names[k]}{suffix}", showlegend=False))

    if phasors is not None: add_arrows(phasors, rms_vals)
    if ref_phasors is not None: add_arrows(ref_phasors, ref_rms, " (Ref)", "dot")

    max_r = max(np.max(rms_vals), np.max(ref_rms) if ref_rms is not None else 0) * 1.1
    if max_r == 0: max_r = 1

    fig.update_layout(title=title, 
        xaxis=dict(range=[-max_r, max_r], showgrid=True, zeroline=True),
        yaxis=dict(range=[-max_r, max_r], showgrid=True, zeroline=True, scaleanchor="x", scaleratio=1),
        height=250, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark")
    return fig

def create_seq_fig(seq_vals, title, ref_seq=None):
    x = ['Pos (+)', 'Neg (-)', 'Zero (0)']
    y = [abs(seq_vals[1]), abs(seq_vals[2]), abs(seq_vals[0])]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=y, name="Atual", marker_color=[THEME['V1'], THEME['V2'], THEME['V0']]))
    if ref_seq is not None:
         y_ref = [abs(ref_seq[1]), abs(ref_seq[2]), abs(ref_seq[0])]
         fig.add_trace(go.Bar(x=x, y=y_ref, name="Ref", opacity=0.5, marker_color=[THEME['V1'], THEME['V2'], THEME['V0']], marker_pattern_shape="/"))
    fig.update_layout(title=title, height=250, margin=dict(l=20, r=20, t=30, b=20), template="plotly_dark")
    return fig

# =========================================================
# 🎬 4. ANIMAÇÃO FLUIDA NATIVA (PARA ABA 2)
# =========================================================
@st.cache_data
def create_fluid_animation(t_vec, i_rms, i_clk, pickup, dial, curve_type):
    # Gera a animação pesada apenas UMA vez e guarda no cache
    total_points = len(t_vec)
    n_frames = 150 
    step = max(1, int(total_points / n_frames))
    indices = range(0, total_points, step)
    
    fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4], vertical_spacing=0.15,
                        subplot_titles=(f"Curva TCC ({curve_type})", "Plano Clarke"))

    cx, cy = get_cached_tcc_curve(pickup, dial, curve_type)
    fig.add_trace(go.Scatter(x=cx, y=cy, mode='lines', line=dict(color='yellow', width=3), name="Curva TCC", hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0.1]*3, y=[0.1]*3, mode='markers+text',
        marker=dict(size=15, color=[THEME['A'], THEME['B'], THEME['C']], symbol=['circle', 'triangle-up', 'square'], line=dict(width=2, color='white')),
        text=["Ia", "Ib", "Ic"], textposition="top right", name="Medição"), row=1, col=1)
    fig.add_vline(x=pickup, line_width=1, line_dash="dash", line_color="gray", row=1, col=1)

    max_clk = np.max(np.abs(i_clk)) if i_clk is not None else 10
    limit = max(10, max_clk * 1.1)
    if i_clk is not None:
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers',
            marker=dict(size=14, color=THEME['V1'], line=dict(width=2, color='white')), name="Clarke"), row=2, col=1)

    frames = []
    for k in indices:
        vals = i_rms[k] if i_rms is not None else [0.1]*3
        tcc_x = [max(v, 0.101) for v in vals]
        tcc_y = [min(calculate_tcc_single(v, pickup, dial, curve_type), 1000) for v in vals]
        
        tcc_text = []
        for v, t_act in zip(vals, tcc_y):
            txt = ""
            if v > 0.5: 
                txt = f"{v:.1f}A"
                if v > pickup: txt += f"<br>{t_act:.2f}s"
            tcc_text.append(txt)

        clk_x = [i_clk[k, 0]] if i_clk is not None else [0]
        clk_y = [i_clk[k, 1]] if i_clk is not None else [0]
        
        frames.append(go.Frame(data=[
            go.Scatter(x=cx, y=cy), 
            go.Scatter(x=tcc_x, y=tcc_y, text=tcc_text),
            go.Scatter(x=clk_x, y=clk_y)
        ], name=f"{t_vec[k]:.3f}"))

    fig.frames = frames
    fig.update_layout(template="plotly_dark", height=750, margin=dict(t=40, b=40),
        xaxis1=dict(type="log", range=[np.log10(0.1), np.log10(30000)], title="Corrente (A)", showgrid=True, gridcolor=THEME['grid']),
        yaxis1=dict(type="log", range=[np.log10(0.01), np.log10(1000)], title="Tempo (s)", showgrid=True, gridcolor=THEME['grid']),
        xaxis2=dict(range=[-limit, limit], title="Alpha", showgrid=True, gridcolor=THEME['grid'], scaleanchor="y2", scaleratio=1),
        yaxis2=dict(range=[-limit, limit], title="Beta", showgrid=True, gridcolor=THEME['grid']),
        updatemenus=[dict(type="buttons", showactive=False, y=1.05, x=1.0, xanchor="right",
            buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=20, redraw=True), fromcurrent=True, mode="immediate")]),
                     dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])])],
        sliders=[dict(steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True))], label=f.name) for f in frames],
            active=0, y=0, x=0.1, len=0.9, currentvalue=dict(prefix="Tempo: ", visible=True), pad=dict(t=20))]
    )
    return fig

# =========================================================
# 🚀 5. CONTROLES E STATE
# =========================================================
if 'data_store' not in st.session_state: st.session_state['data_store'] = {}
if 'ref_data' not in st.session_state: st.session_state['ref_data'] = None
if 'idx' not in st.session_state: st.session_state['idx'] = 0
if 'playing' not in st.session_state: st.session_state['playing'] = False

with st.sidebar:
    st.header("⚡ T2F Master")
    uploaded_files = st.file_uploader("Arquivos .mat", type=['mat'], accept_multiple_files=True)
    if uploaded_files:
        for f in uploaded_files:
            if f.name not in st.session_state['data_store']:
                try:
                    raw = sio.loadmat(f, squeeze_me=True)
                    st.session_state['data_store'][f.name] = parse_mat_file(raw)
                except: pass

    opts = list(st.session_state['data_store'].keys())
    selected_file = st.selectbox("Arquivo", opts) if opts else None
    
    # Referência
    current_data = st.session_state['data_store'].get(selected_file)
    c1, c2 = st.columns(2)
    if c1.button("Fixar Ref.") and current_data: st.session_state['ref_data'] = current_data
    if c2.button("Limpar Ref."): st.session_state['ref_data'] = None

    # Sinais
    available_keys = []
    if current_data:
        available_keys = [k for k in current_data.keys() if k != 't']
        available_keys.sort()
    
    def_v = next((k for k in available_keys if 'V' in k), None)
    def_i = next((k for k in available_keys if 'I' in k), None)
    sel_v = st.selectbox("Tensão", available_keys, index=available_keys.index(def_v) if def_v else 0) if available_keys else None
    sel_i = st.selectbox("Corrente", available_keys, index=available_keys.index(def_i) if def_i else 0) if available_keys else None

    # --- PLAYER (CONTROLA ABAS 1 E 3) ---
    st.divider()
    st.markdown("#### ⏯️ Player (Abas 1 e 3)")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    if col_p1.button("▶", help="Play"):
        st.session_state['playing'] = True
        st.rerun()
    if col_p2.button("⏸", help="Pause"):
        st.session_state['playing'] = False
        st.rerun()
    if col_p3.button("⏹", help="Reset"):
        st.session_state['playing'] = False
        st.session_state['idx'] = 0
        st.rerun()
        
    t_max = len(current_data['t']) - 1 if current_data else 100
    idx_val = st.slider("Tempo", 0, t_max, st.session_state['idx'], label_visibility="collapsed")
    if idx_val != st.session_state['idx']:
        st.session_state['idx'] = idx_val # Atualiza se o usuário mexer no slider
    
    speed_step = st.number_input("Velocidade (Step)", 1, 100, 2)

    # Proteção
    st.divider()
    st.subheader("🛡️ Config. Proteção")
    curve_type = st.selectbox("Curva", list(CURVES.keys()))
    pickup = st.number_input("Pickup (A)", value=25.0)
    dial = st.number_input("Dial", value=0.5)

# --- CORPO PRINCIPAL ---
if not current_data or not sel_v or not sel_i:
    st.info("Carregue arquivos e selecione sinais.")
    st.stop()

# Dados e Índices
t_vec = current_data['t']
curr_idx = st.session_state['idx']
time_curr = t_vec[curr_idx]

v_data = current_data[sel_v]; i_data = current_data[sel_i]
v_rms = v_data.get('rms'); i_rms = i_data.get('rms')
i_clk = i_data.get('clarke')
v_ph = v_data.get('phasor'); i_ph = i_data.get('phasor')
v_seq = v_data.get('seq'); i_seq = i_data.get('seq')

# Referência
ref_v_rms, ref_i_rms, ref_t = None, None, None
ref_v_ph, ref_i_ph, ref_v_seq, ref_i_seq = None, None, None, None
if st.session_state['ref_data']:
    rd = st.session_state['ref_data']; ref_t = rd['t']
    ref_idx = min(curr_idx, len(ref_t)-1)
    if sel_v in rd: ref_v_rms = rd[sel_v].get('rms'); ref_v_ph = rd[sel_v].get('phasor'); ref_v_seq = rd[sel_v].get('seq')
    if sel_i in rd: ref_i_rms = rd[sel_i].get('rms'); ref_i_ph = rd[sel_i].get('phasor'); ref_i_seq = rd[sel_i].get('seq')

# --- ABAS ---
tab1, tab2, tab3 = st.tabs(["📊 Análise", "🛡️ Proteção (Fluida)", "🔁 Comparação"])

# ABA 1: ANÁLISE (CONTROLADA PELO PLAYER PYTHON)
with tab1:
    st.markdown(f"**Tempo:** `{time_curr:.4f}s` (Use o player lateral)")
    c1, c2 = st.columns(2)
    c1.plotly_chart(create_waveform_fig(t_vec, v_rms, "Tensão (RMS)", "V", time_curr), use_container_width=True)
    c2.plotly_chart(create_waveform_fig(t_vec, i_rms, "Corrente (RMS)", "A", time_curr), use_container_width=True)
    
    val_v_ph = v_ph[curr_idx] if v_ph is not None else [0]*3
    val_i_ph = i_ph[curr_idx] if i_ph is not None else [0]*3
    val_v_now = v_rms[curr_idx] if v_rms is not None else [0]*3
    val_i_now = i_rms[curr_idx] if i_rms is not None else [0]*3
    
    c3, c4 = st.columns(2)
    c3.plotly_chart(create_phasor_fig(val_v_ph, val_v_now, "Fasores Tensão"), use_container_width=True)
    c4.plotly_chart(create_phasor_fig(val_i_ph, val_i_now, "Fasores Corrente"), use_container_width=True)
    
    val_v_seq = v_seq[curr_idx] if v_seq is not None else [0]*3
    val_i_seq = i_seq[curr_idx] if i_seq is not None else [0]*3
    c5, c6 = st.columns(2)
    c5.plotly_chart(create_seq_fig(val_v_seq, "Sequência Tensão"), use_container_width=True)
    c6.plotly_chart(create_seq_fig(val_i_seq, "Sequência Corrente"), use_container_width=True)

# ABA 2: PROTEÇÃO (ANIMAÇÃO FLUIDA NATIVA)
with tab2:
    st.caption("ℹ️ Esta aba usa um player interno independente para garantir 60 FPS.")
    with st.spinner("Preparando animação fluida..."):
        fig_anim = create_fluid_animation(t_vec, i_rms, i_clk, pickup, dial, curve_type)
        st.plotly_chart(fig_anim, use_container_width=True)

# ABA 3: COMPARAÇÃO (CONTROLADA PELO PLAYER PYTHON)
with tab3:
    st.markdown(f"**Tempo:** `{time_curr:.4f}s` (Use o player lateral)")
    if st.session_state['ref_data'] is None:
        st.warning("Defina uma Referência no painel lateral.")
    else:
        k1, k2 = st.columns(2)
        k1.plotly_chart(create_waveform_fig(t_vec, v_rms, "Comp V", "V", time_curr, ref_v_rms, ref_t), use_container_width=True)
        k2.plotly_chart(create_waveform_fig(t_vec, i_rms, "Comp I", "A", time_curr, ref_i_rms, ref_t), use_container_width=True)
        
        # Dados Ref
        rvp = ref_v_ph[ref_idx] if ref_v_ph is not None else [0]*3
        rip = ref_i_ph[ref_idx] if ref_i_ph is not None else [0]*3
        rvn = ref_v_rms[ref_idx] if ref_v_rms is not None else [0]*3
        rin = ref_i_rms[ref_idx] if ref_i_rms is not None else [0]*3

        k3, k4 = st.columns(2)
        k3.plotly_chart(create_phasor_fig(val_v_ph, val_v_now, "Comp Fasor V", rvp, rvn), use_container_width=True)
        k4.plotly_chart(create_phasor_fig(val_i_ph, val_i_now, "Comp Fasor I", rip, rin), use_container_width=True)
        
        rvs = ref_v_seq[ref_idx] if ref_v_seq is not None else [0]*3
        ris = ref_i_seq[ref_idx] if ref_i_seq is not None else [0]*3
        k5, k6 = st.columns(2)
        k5.plotly_chart(create_seq_fig(val_v_seq, "Comp Seq V", rvs), use_container_width=True)
        k6.plotly_chart(create_seq_fig(val_i_seq, "Comp Seq I", ris), use_container_width=True)

# =========================================================
# 🔄 LOOP DE ANIMAÇÃO DO PLAYER PYTHON
# =========================================================
if st.session_state['playing']:
    if st.session_state['idx'] < len(t_vec) - 1:
        st.session_state['idx'] += speed_step
        st.rerun() # Atualiza a tela para o próximo frame
    else:
        st.session_state['playing'] = False
        st.rerun()