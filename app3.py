import streamlit as st
import numpy as np
import scipy.io as sio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# ⚙️ 1. CONFIGURAÇÕES VISUAIS
# =========================================================
st.set_page_config(page_title="Comparison Studio", layout="wide", page_icon="⚖️")

THEME = {
    "A": "#00ffff", "B": "#ff3333", "C": "#00ff00", # Cores Sinais Atuais
    "RA": "#008888", "RB": "#883333", "RC": "#008800", # Cores Referência (Mais escuras)
    "V1": "#4facfe", "V2": "#f093fb", "V0": "#fcc203",
    "grid": "#333"
}

CURVES = {
    "IEC Standard Inverse":  (0.14, 0.0, 0.02),
    "IEC Very Inverse":      (13.5, 0.0, 1.0),
    "IEC Extremely Inverse": (80.0, 0.0, 2.0),
    "IEC Long Time Inverse": (120.0, 0.0, 1.0),
    "IEEE Moderately Inv":   (0.0515, 0.114, 0.02),
    "IEEE Very Inverse":     (19.61, 0.491, 2.0),
    "IEEE Extremely Inv":    (28.2, 0.1217, 2.0)
}

# =========================================================
# 🧮 2. CÁLCULOS
# =========================================================
@st.cache_data
def get_tcc_curve(pickup, dial, curve_name):
    """Gera a linha estática da curva de proteção"""
    i_plot = np.logspace(np.log10(0.1), np.log10(30000), 400)
    if curve_name not in CURVES: params = CURVES["IEC Standard Inverse"]
    else: params = CURVES[curve_name]
    A, B, p = params
    
    safe_Ip = pickup if pickup > 0 else 0.001
    M = i_plot / safe_Ip
    t_plot = np.full_like(i_plot, 2000.0, dtype=float)
    mask = M > 1.001
    if np.any(mask):
        denom = np.power(M[mask], p) - 1
        denom[denom == 0] = 1e-9
        t_plot[mask] = dial * ((A / denom) + B)
    return i_plot, t_plot

def calc_trip_time(I, Ip, TD, curve_name):
    """Calcula o tempo de atuação para um ponto específico"""
    if curve_name not in CURVES: params = CURVES["IEC Standard Inverse"]
    else: params = CURVES[curve_name]
    A, B, p = params
    M = I / (Ip if Ip > 0 else 0.001)
    if M <= 1.001: return 1000.0
    val = TD * ((A / ((M**p)-1)) + B)
    return min(val, 1000.0)

def parse_mat_file(mat_data):
    """Lê o arquivo .mat e organiza os dados"""
    parsed = {}
    # Tenta encontrar vetor de tempo
    t = mat_data.get('t') if 't' in mat_data else mat_data.get('time')
    if t is None: return None
    parsed['t'] = t.flatten()
    
    for key in mat_data.keys():
        if key.startswith('__') or key in ['t', 'time']: continue
        
        # Identifica o nome base do sinal (ex: 'VI_I_A' de 'VI_I_A_rms')
        base = key.replace('_rms','').replace('_phasor','').replace('_seq','').replace('_clarke','').replace('_raw','')
        
        tipo = 'raw'
        if '_rms' in key: tipo = 'rms'
        elif '_phasor' in key: tipo = 'phasor'
        elif '_seq' in key: tipo = 'seq'
        elif '_clarke' in key: tipo = 'clarke'
        
        if base not in parsed: parsed[base] = {}
        parsed[base][tipo] = mat_data[key]
    return parsed

# =========================================================
# 🎬 3. MOTOR GRÁFICO (COMPARISON ENGINE)
# =========================================================
def create_comparison_dashboard(t, v_curr, i_curr, v_ref, i_ref, pickup, dial, curve):
    
    # --- PREPARAÇÃO DE DADOS ---
    def get_d(d_dict, key, shape_def):
        if d_dict is None: return np.zeros(shape_def)
        return d_dict.get(key, np.zeros(shape_def))

    N = len(t)
    # Dados Atuais
    vc_rms = get_d(v_curr, 'rms', (N,3))
    ic_rms = get_d(i_curr, 'rms', (N,3))
    vc_ph = get_d(v_curr, 'phasor', (N,3))
    ic_ph = get_d(i_curr, 'phasor', (N,3))
    ic_clk = get_d(i_curr, 'clarke', (N,2))
    
    # Dados de Referência (Se houver)
    has_ref = v_ref is not None
    vr_rms = get_d(v_ref, 'rms', (N,3))
    ir_rms = get_d(i_ref, 'rms', (N,3))
    vr_ph = get_d(v_ref, 'phasor', (N,3))
    ir_ph = get_d(i_ref, 'phasor', (N,3))

    # Downsampling para animação (Performance)
    n_frames = 120
    step = max(1, int(N / n_frames))
    indices = range(0, N, step)

    # --- LAYOUT GRID ---
    # Linha 1: Comparação Ondas (V e I) | TCC (Direita)
    # Linha 2: Fasores (V e I) | TCC (Continuação)
    # Linha 3: Clarke (Ocupa largura total)
    
    fig = make_subplots(
        rows=3, cols=3,
        specs=[
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy", "rowspan": 2}], # TCC ocupa 2 linhas na direita
            [{"type": "xy"}, {"type": "xy"}, None],
            [{"type": "xy", "colspan": 3}, None, None] # Clarke largo embaixo
        ],
        column_widths=[0.3, 0.3, 0.4],
        subplot_titles=("Comparação Tensão (V)", "Comparação Corrente (A)", "Proteção (TCC)", 
                        "Fasores V (Ref=Sombra)", "Fasores I (Ref=Sombra)", "Plano Alpha-Beta (Clarke)")
    )

    # --- ELEMENTOS ESTÁTICOS (FUNDO) ---
    ds = 20 # Downsample visual estático
    
    # 1. Ondas (Fundo)
    for i, c in enumerate([THEME['A'], THEME['B'], THEME['C']]):
        # Atual (Linha contínua)
        fig.add_trace(go.Scatter(x=t[::ds], y=vc_rms[::ds,i], line=dict(color=c, width=1), opacity=0.4, showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=t[::ds], y=ic_rms[::ds,i], line=dict(color=c, width=1), opacity=0.4, showlegend=False), row=1, col=2)
        # Ref (Linha tracejada)
        if has_ref:
            fig.add_trace(go.Scatter(x=t[::ds], y=vr_rms[::ds,i], line=dict(color=c, width=1, dash='dot'), opacity=0.3, showlegend=False), row=1, col=1)
            fig.add_trace(go.Scatter(x=t[::ds], y=ir_rms[::ds,i], line=dict(color=c, width=1, dash='dot'), opacity=0.3, showlegend=False), row=1, col=2)

    # 2. Inicialização dos Pontos Móveis (Trace Placeholders)
    # Precisamos criar os traces na ordem exata que vamos atualizar nos frames
    
    # [0-1] Pontos Ondas Atuais
    fig.add_trace(go.Scatter(x=[t[0]]*3, y=[0]*3, mode='markers', marker=dict(color=[THEME['A'], THEME['B'], THEME['C']], size=8)), row=1, col=1)
    fig.add_trace(go.Scatter(x=[t[0]]*3, y=[0]*3, mode='markers', marker=dict(color=[THEME['A'], THEME['B'], THEME['C']], size=8)), row=1, col=2)
    
    # [2-3] Pontos Ondas Ref (Se houver)
    if has_ref:
        fig.add_trace(go.Scatter(x=[t[0]]*3, y=[0]*3, mode='markers', marker=dict(color=[THEME['A'], THEME['B'], THEME['C']], size=6, symbol='circle-open')), row=1, col=1)
        fig.add_trace(go.Scatter(x=[t[0]]*3, y=[0]*3, mode='markers', marker=dict(color=[THEME['A'], THEME['B'], THEME['C']], size=6, symbol='circle-open')), row=1, col=2)

    # [4-5] Fasores Ref (Sombra)
    if has_ref:
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(width=1, dash='dot', color='gray'), marker=dict(symbol='x')), row=2, col=1)
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(width=1, dash='dot', color='gray'), marker=dict(symbol='x')), row=2, col=2)

    # [6-7] Fasores Atuais
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(width=3)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', line=dict(width=3)), row=2, col=2)

    # [8-10] TCC (Curva + Pontos)
    cx, cy = get_tcc_curve(pickup, dial, curve)
    fig.add_trace(go.Scatter(x=cx, y=cy, line=dict(color='yellow', width=3), name="Curva TCC"), row=1, col=3) # Estático
    fig.add_vline(x=pickup, line_dash="dash", line_color="gray", row=1, col=3) # Estático
    
    fig.add_trace(go.Scatter(x=[0.1]*3, y=[0.1]*3, mode='markers+text', marker=dict(color=[THEME['A'], THEME['B'], THEME['C']], size=12, line=dict(width=1, color='white')), name="Atual"), row=1, col=3)
    if has_ref:
        fig.add_trace(go.Scatter(x=[0.1]*3, y=[0.1]*3, mode='markers', marker=dict(color=[THEME['A'], THEME['B'], THEME['C']], size=10, symbol='circle-open'), name="Ref"), row=1, col=3)

    # [11] Clarke
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(color='white', size=10, line=dict(width=2, color='cyan'))), row=3, col=1)

    # --- GERAÇÃO DE FRAMES ---
    frames = []
    
    # Escalas para Eixos Fixos
    max_v = max(1, np.max(vc_rms)*1.1)
    max_i = max(1, np.max(ic_rms)*1.1)
    max_clk = max(10, np.max(np.abs(ic_clk))*1.1)

    for k in indices:
        tc = t[k]
        
        # Dados do frame
        vc = vc_rms[k]; ic = ic_rms[k]
        vr = vr_rms[k] if has_ref else [0]*3; ir = ir_rms[k] if has_ref else [0]*3
        
        # Helper Fasores
        def get_vecs(phasors):
            x, y = [], []
            for p in phasors:
                mag = np.abs(p); ang = np.angle(p)
                x.extend([0, mag*np.cos(ang), None])
                y.extend([0, mag*np.sin(ang), None])
            return x, y
        
        vc_x, vc_y = get_vecs(vc_ph[k]); ic_x, ic_y = get_vecs(ic_ph[k])
        vr_x, vr_y = get_vecs(vr_ph[k]) if has_ref else ([],[]); ir_x, ir_y = get_vecs(ir_ph[k]) if has_ref else ([],[])

        # TCC Calc
        tcc_curr_x = [max(v, 0.101) for v in ic]
        tcc_curr_y = [calc_trip_time(v, pickup, dial, curve) for v in ic]
        tcc_curr_txt = [f"{v:.1f}A" if v > 0.5 else "" for v in ic]
        
        tcc_ref_x = [max(v, 0.101) for v in ir]
        tcc_ref_y = [calc_trip_time(v, pickup, dial, curve) for v in ir]

        # DATA LIST DO FRAME (Deve bater com a ordem dos traces acima)
        data_list = []
        
        # 1. Pontos Onda Atual
        data_list.append(go.Scatter(x=[tc]*3, y=vc))
        data_list.append(go.Scatter(x=[tc]*3, y=ic))
        
        # 2. Pontos Onda Ref
        if has_ref:
            data_list.append(go.Scatter(x=[tc]*3, y=vr))
            data_list.append(go.Scatter(x=[tc]*3, y=ir))

        # 3. Fasores Ref
        if has_ref:
            data_list.append(go.Scatter(x=vr_x, y=vr_y))
            data_list.append(go.Scatter(x=ir_x, y=ir_y))
            
        # 4. Fasores Atuais
        data_list.append(go.Scatter(x=vc_x, y=vc_y))
        data_list.append(go.Scatter(x=ic_x, y=ic_y))
        
        # 5. TCC (Pula estáticos, atualiza dinâmicos)
        data_list.append(go.Scatter()) # Curva estática (sem mudança)
        data_list.append(go.Scatter(x=tcc_curr_x, y=tcc_curr_y, text=tcc_curr_txt)) # Atual
        if has_ref:
            data_list.append(go.Scatter(x=tcc_ref_x, y=tcc_ref_y)) # Ref

        # 6. Clarke
        data_list.append(go.Scatter(x=[ic_clk[k,0]], y=[ic_clk[k,1]]))

        frames.append(go.Frame(data=data_list, name=f"{tc:.3f}"))

    fig.frames = frames

    # --- LAYOUT FINAL ---
    fig.update_layout(
        template="plotly_dark", height=850,
        margin=dict(l=20, r=20, t=50, b=50),
        # Eixos Ondas
        xaxis1=dict(range=[0, t[-1]]), yaxis1=dict(range=[0, max_v]),
        xaxis2=dict(range=[0, t[-1]]), yaxis2=dict(range=[0, max_i]),
        # Eixos Fasores (Quadrados)
        xaxis4=dict(range=[-max_v, max_v]), yaxis4=dict(range=[-max_v, max_v], scaleanchor="x4", scaleratio=1),
        xaxis5=dict(range=[-max_i, max_i]), yaxis5=dict(range=[-max_i, max_i], scaleanchor="x5", scaleratio=1),
        # Eixo TCC (Log)
        xaxis3=dict(type='log', range=[np.log10(0.1), np.log10(30000)], title="Corrente (A)"),
        yaxis3=dict(type='log', range=[np.log10(0.01), np.log10(1000)], title="Tempo (s)"),
        # Eixo Clarke (Quadrado)
        xaxis7=dict(range=[-max_clk, max_clk], title="Alpha"), 
        yaxis7=dict(range=[-max_clk, max_clk], title="Beta", scaleanchor="x7", scaleratio=1),
        
        # Controles
        updatemenus=[dict(type="buttons", showactive=False, x=0.5, y=-0.1, xanchor="center", direction="left",
            buttons=[dict(label="▶ Play", method="animate", args=[None, dict(frame=dict(duration=20, redraw=True), fromcurrent=True, mode="immediate")]),
                     dict(label="⏸ Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(steps=[dict(method='animate', args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True))], label=f.name) for f in frames],
            active=0, x=0.1, len=0.8, pad=dict(t=20), currentvalue=dict(prefix="Tempo: ", visible=True))]
    )
    return fig

# =========================================================
# 🚀 4. APLICAÇÃO
# =========================================================
st.title("⚖️ Comparison Studio & TCC")

with st.sidebar:
    st.header("1. Arquivos")
    uploaded = st.file_uploader("Carregar .MAT", type=['mat'], accept_multiple_files=True)
    
    if uploaded:
        if 'db' not in st.session_state: st.session_state['db'] = {}
        for f in uploaded:
            # Lê o arquivo
            raw = sio.loadmat(f, squeeze_me=True)
            # Armazena parseado no banco
            st.session_state['db'][f.name] = parse_mat_file(raw)
    
    opts = list(st.session_state.get('db', {}).keys())
    
    st.divider()
    f_curr = st.selectbox("Arquivo Principal (Atual)", opts) if opts else None
    f_ref = st.selectbox("Arquivo Referência (Opcional)", ["Nenhum"] + opts) if opts else None
    
    st.divider()
    st.header("2. Proteção")
    curve = st.selectbox("Curva TCC", list(CURVES.keys()))
    pickup = st.number_input("Pickup", 25.0)
    dial = st.number_input("Dial", 0.5)
    
    btn = st.button("Gerar Comparação Fluida", type="primary")

if btn and f_curr:
    with st.spinner("Construindo Dashboard de Comparação..."):
        db = st.session_state['db']
        d_curr = db[f_curr]
        d_ref = db[f_ref] if f_ref != "Nenhum" else None
        
        # Auto-detect keys
        ks = list(d_curr.keys())
        v_key = next((k for k in ks if 'V' in k), ks[0])
        i_key = next((k for k in ks if 'I' in k), ks[0])
        
        v_ref_data = d_ref[v_key] if d_ref and v_key in d_ref else None
        i_ref_data = d_ref[i_key] if d_ref and i_key in d_ref else None
        
        # --- CORREÇÃO APLICADA AQUI ---
        fig = create_comparison_dashboard(
            d_curr['t'], # Passa o array de tempo direto (Correção do erro anterior)
            d_curr[v_key], d_curr[i_key],
            v_ref_data, i_ref_data,
            pickup, dial, curve
        )
        st.plotly_chart(fig, use_container_width=True)

elif not f_curr:
    st.info("Carregue arquivos para começar.")