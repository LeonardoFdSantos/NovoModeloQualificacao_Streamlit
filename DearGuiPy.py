import dearpygui.dearpygui as dpg
import numpy as np
import scipy.io as sio
import scipy.ndimage
import h5py
import os
import warnings
import traceback

warnings.filterwarnings("ignore")

dpg.create_context()

# ==============================================================================
# 1. PROCESSADOR DE SINAIS
# ==============================================================================
class SignalProcessor:
    def __init__(self, t, v_matrix, i_matrix, freq=60):
        t = np.array(t, dtype=np.float64)
        v_matrix = np.array(v_matrix, dtype=np.float64)
        i_matrix = np.array(i_matrix, dtype=np.float64)

        self.t = t.flatten()
        self.freq = freq
        
        self.v_raw = self._fix_shape(v_matrix, len(self.t))
        self.i_raw = self._fix_shape(i_matrix, len(self.t))
        
        L = min(len(self.t), len(self.v_raw), len(self.i_raw))
        if L < 2: raise ValueError("Dados vazios.")
        
        self.t = self.t[:L]
        self.v_raw = self.v_raw[:L]
        self.i_raw = self.i_raw[:L]
        
        dt = self.t[1] - self.t[0]
        self.samples = int(1.0 / (freq * dt)) if dt > 0 else 1
        if self.samples < 1: self.samples = 1
        
        print(f"Processando {L} pts. Janela {self.samples}")

        self.v_rms = np.nan_to_num(self._rms(self.v_raw))
        self.i_rms = np.nan_to_num(self._rms(self.i_raw))
        
        clk_i = self._clarke(self.i_raw)
        self.i_clarke = {k: np.nan_to_num(v) for k, v in clk_i.items()}
        
        self.i_seq = np.nan_to_num(self._seq(self.i_raw))

    def _fix_shape(self, mat, target_len):
        if mat.ndim == 1:
            col = mat.flatten()
            if len(col) == 0: return np.zeros((target_len, 3))
            return np.stack([col, col, col], axis=1)
        if mat.ndim == 2:
            if mat.shape[0] == 3 and mat.shape[1] > 3: return mat.T
            if mat.shape[1] == 3: return mat
        return np.zeros((target_len, 3))

    def _rms(self, x):
        return np.sqrt(np.abs(scipy.ndimage.uniform_filter1d(x**2, self.samples, axis=0)))

    def _clarke(self, abc):
        a, b, c = abc[:,0], abc[:,1], abc[:,2]
        alpha = (2*a - b - c)/3.0
        beta  = (b - c)/np.sqrt(3.0)
        return {'alpha': alpha, 'beta': beta}

    def _seq(self, x):
        rot = np.exp(-1j * 2 * np.pi * self.freq * self.t)
        ph = np.zeros_like(x, dtype=complex)
        for k in range(3):
            ph[:,k] = scipy.ndimage.uniform_filter1d(x[:,k] * rot, self.samples) * np.sqrt(2)
        a = np.exp(1j * 2*np.pi/3)
        v0 = (ph[:,0] + ph[:,1] + ph[:,2])/3.0
        v1 = (ph[:,0] + a*ph[:,1] + a**2*ph[:,2])/3.0
        v2 = (ph[:,0] + a**2*ph[:,1] + a*ph[:,2])/3.0
        return np.stack([np.abs(v0), np.abs(v1), np.abs(v2)], axis=1)

class AppState:
    def __init__(self):
        self.proc_curr = None; self.proc_ref = None
        self.idx = 0; self.playing = False; self.speed = 5
        self.pickup = 25.0; self.dial = 0.5; self.curve_idx = 0
        self.loaded_raw = None

state = AppState()

CURVES = {
    "IEC Standard Inverse":  (0.14, 0.0, 0.02),
    "IEC Very Inverse":      (13.5, 0.0, 1.0),
    "IEC Extremely Inverse": (80.0, 0.0, 2.0),
    "IEC Long Time Inverse": (120.0, 0.0, 1.0),
    "IEEE Moderately Inverse": (0.0515, 0.1140, 0.02),
    "IEEE Very Inverse":       (19.61, 0.4910, 2.0),
    "IEEE Extremely Inverse":  (28.2, 0.1217, 2.0),
    "US CO8 Inverse":          (5.95, 0.180, 2.0),
    "US CO2 Short Time":       (0.02394, 0.01694, 0.02)
}
CURVE_LIST = list(CURVES.keys())

# CORES
C_A = (0, 255, 255, 255)   # Cyan
C_B = (255, 50, 50, 255)   # Red
C_C = (0, 255, 0, 255)     # Green
C_SEQ = [(255, 215, 0, 255), (0, 191, 255, 255), (255, 0, 255, 255)]

# ==============================================================================
# 2. CARREGAMENTO
# ==============================================================================
def load_data(path):
    try: 
        data = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                try: data[k] = f[k][()]
                except: pass
        return data
    except:
        try: return sio.loadmat(path, squeeze_me=True)
        except: return None

def load_file_cb(sender, app_data, user_data):
    try:
        path = list(app_data['selections'].values())[0]
        raw = load_data(path)
        if not raw: return

        keys = [k for k in raw.keys() if not k.startswith('_') and k not in ['t', 'time', 'm1']]
        keys.sort()

        def find_best(terms):
            for k in keys:
                if all(t in k for t in terms): return k
            return None

        v_guess = find_best(['V', 'raw']) or find_best(['V']) or (keys[0] if keys else "")
        i_guess = find_best(['I', 'raw']) or find_best(['I']) or (keys[1] if len(keys)>1 else "")

        if user_data == "curr":
            state.loaded_raw = raw
            dpg.configure_item("c_v_mat", items=keys)
            dpg.configure_item("c_i_mat", items=keys)
            dpg.set_value("c_v_mat", v_guess)
            dpg.set_value("c_i_mat", i_guess)
            dpg.set_value("status", "Carregado (Principal).")
            
        elif user_data == "ref":
            t_key = next((k for k in raw.keys() if k.lower() in ['t', 'time']), None)
            if t_key and v_guess and i_guess:
                try:
                    proc = SignalProcessor(raw[t_key], raw[v_guess], raw[i_guess])
                    state.proc_ref = proc
                    setup_plots(proc, "ref")
                    dpg.set_value("status", f"Ref: {os.path.basename(path)}")
                except: pass

    except: dpg.set_value("status", "Erro Load")

def trigger_process():
    try:
        if not state.loaded_raw: return
        v_key = dpg.get_value("c_v_mat")
        i_key = dpg.get_value("c_i_mat")
        raw = state.loaded_raw
        t_key = next((k for k in raw.keys() if k.lower() in ['t', 'time']), None)
        if not t_key: return
            
        proc = SignalProcessor(raw[t_key], raw[v_key], raw[i_key])
        state.proc_curr = proc; state.idx = 0
        
        dpg.configure_item("timeline", max_value=len(proc.t)-1)
        dpg.set_value("timeline", 0)
        
        setup_plots(proc, "curr")
        dpg.set_value("status", "Processado!")
        
    except Exception as e:
        dpg.set_value("status", f"Erro: {str(e)}")

# ==============================================================================
# 3. PLOTS & VISUALS
# ==============================================================================
def cb_timeline(sender, app_data):
    state.idx = int(app_data)
    state.playing = False
    update_visuals()

def setup_plots(p, target):
    suf = "" if target == "curr" else "_ref"
    
    # --- CONFIGURAÇÃO DE CORES E LEGENDAS ---
    if target == "curr":
        # Atual: Opaco, Linha Grossa
        colors = [C_A, C_B, C_C]
        thick = 2.0
        op = 255
        lbl_suf = "" # Sem sufixo
    else:
        # Ref: Transparente (Mesma cor), Linha Fina
        # Alpha 120 = Semi transparente
        colors = [list(c[:3])+[120] for c in [C_A, C_B, C_C]]
        thick = 1.0
        op = 120
        lbl_suf = " (Ref)"

    t = p.t.tolist()
    
    def line(tag_base, y_arr, col_idx, label_name):
        tag = tag_base + suf
        col = colors[col_idx]
        
        # Mapeamento
        if "rms_v" in tag_base: parent = "ax_rms_yv"
        elif "rms_i" in tag_base: parent = "ax_rms_yi"
        elif "clk_time" in tag_base: parent = "ax_clk_time_y"
        elif "xy_full" in tag_base: parent = "ax_clk_xy_y"
        else: return

        if len(y_arr) != len(t): return
        y_list = y_arr.tolist()

        if dpg.does_item_exist(tag): dpg.set_value(tag, [t, y_list])
        else:
            if dpg.does_item_exist(parent):
                # Se for Clarke XY (Rastro)
                if "xy_full" in tag_base:
                    dpg.add_line_series(p.i_clarke['alpha'].tolist(), p.i_clarke['beta'].tolist(), 
                                        parent=parent, tag=tag, label=label_name + lbl_suf)
                else:
                    dpg.add_line_series(t, y_list, parent=parent, tag=tag, label=label_name + lbl_suf)
                
                with dpg.theme() as tm:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, thick, category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(tag, tm)

    try:
        # RMS (A, B, C)
        line("rms_va", p.v_rms[:,0], 0, "Va"); line("rms_vb", p.v_rms[:,1], 1, "Vb"); line("rms_vc", p.v_rms[:,2], 2, "Vc")
        line("rms_ia", p.i_rms[:,0], 0, "Ia"); line("rms_ib", p.i_rms[:,1], 1, "Ib"); line("rms_ic", p.i_rms[:,2], 2, "Ic")
        
        # Clarke Tempo (A, B)
        line("clk_time_a", p.i_clarke['alpha'], 0, "Alpha")
        line("clk_time_b", p.i_clarke['beta'], 1, "Beta")
        
        # Clarke XY (Rastro)
        # Se for Ref, plota o rastro completo em cor única (Cyan transparente) para comparar
        if target == "ref":
            line("xy_full", p.i_clarke['beta'], 0, "Rastro") 

        if target == "curr":
            for ax in ["ax_rms_x", "ax_rms_yv", "ax_rms_x_i", "ax_rms_yi", "ax_clk_t", "ax_clk_time_y"]:
                dpg.fit_axis_data(ax)
            dpg.set_value("bar_seq", [[1, 2, 3], [0, 0, 0]])
            dpg.fit_axis_data("ax_seq_y")

        update_tcc()
    except: pass

def update_visuals():
    if not state.proc_curr: return
    try:
        if state.playing:
            state.idx += state.speed
            if state.idx >= len(state.proc_curr.t): state.idx = 0
            dpg.set_value("timeline", int(state.idx))
        
        idx = int(state.idx); t = state.proc_curr.t[idx]
        total = len(state.proc_curr.t)
        
        dpg.set_value("txt_info", f"T: {t:.4f}s")
        dpg.set_value("txt_sample", f"Pt: {idx}/{total}")
        
        # Cursores
        for c in ["cur_rms", "cur_rms_i", "cur_clk_t"]: 
            if dpg.does_item_exist(c): dpg.set_value(c, t)
            
        # Seq
        seq_vals = state.proc_curr.i_seq[idx].tolist()
        dpg.set_value("bar_seq", [[1, 2, 3], seq_vals])
        
        if state.proc_ref and len(state.proc_ref.i_seq) > idx:
            seq_ref = state.proc_ref.i_seq[idx].tolist()
            dpg.set_value("bar_seq_ref", [[1.25, 2.25, 3.25], seq_ref])
        
        # Clarke XY (Dinâmico)
        a = state.proc_curr.i_clarke['alpha']; b = state.proc_curr.i_clarke['beta']
        s = max(0, idx-300)
        dpg.set_value("tr_clk", [a[s:idx].tolist(), b[s:idx].tolist()])
        dpg.set_value("dot_clk", [[a[idx]], [b[idx]]])
        
        # TCC
        i_inst = state.proc_curr.i_rms[idx]
        trips = [calc_trip(v) for v in i_inst]
        dpg.set_value("dot_tcc_a", [[i_inst[0]], [trips[0]]])
        dpg.set_value("dot_tcc_b", [[i_inst[1]], [trips[1]]])
        dpg.set_value("dot_tcc_c", [[i_inst[2]], [trips[2]]])
        
        if state.proc_ref and len(state.proc_ref.i_rms) > idx:
            i_ref = state.proc_ref.i_rms[idx]
            trip_ref = [calc_trip(v) for v in i_ref]
            dpg.set_value("dot_tcc_a_ref", [[i_ref[0]], [trip_ref[0]]])
            dpg.set_value("dot_tcc_b_ref", [[i_ref[1]], [trip_ref[1]]])
            dpg.set_value("dot_tcc_c_ref", [[i_ref[2]], [trip_ref[2]]])
        
    except: pass

def calc_trip(I):
    Ip = state.pickup; TD = state.dial
    if I < Ip * 1.001: return 1000.0
    A, B, p = CURVES[list(CURVES.keys())[state.curve_idx]]
    try: 
        val = TD * ( (A / ((I/Ip)**p - 1)) + B )
        return max(0.0001, min(1000.0, val))
    except: return 1000.0

def update_tcc():
    x = np.logspace(0, 4, 200)
    y = [calc_trip(v) for v in x]
    dpg.set_value("ser_tcc", [x.tolist(), y])

def cb_tcc(s, a):
    if s == "in_pk": state.pickup = a
    if s == "in_td": state.dial = a
    if s == "in_cv": state.curve_idx = CURVE_LIST.index(a)
    update_tcc()

# ==============================================================================
# GUI
# ==============================================================================
with dpg.file_dialog(directory_selector=False, show=False, callback=load_file_cb, id="fd_curr", width=600, height=400, user_data="curr"):
    dpg.add_file_extension(".mat", color=(0, 255, 0))
with dpg.file_dialog(directory_selector=False, show=False, callback=load_file_cb, id="fd_ref", width=600, height=400, user_data="ref"):
    dpg.add_file_extension(".mat", color=(255, 255, 0))

with dpg.window(tag="Win"):
    with dpg.collapsing_header(label="Controles", default_open=True):
        with dpg.group(horizontal=True):
            dpg.add_button(label="1. Carregar .MAT", callback=lambda: dpg.show_item("fd_curr"))
            dpg.add_button(label="2. Ref", callback=lambda: dpg.show_item("fd_ref"))
            dpg.add_text("...", tag="status", color=(0,255,255))
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("Tensão"); dpg.add_combo([], tag="c_v_mat", width=200)
            with dpg.group():
                dpg.add_text("Corrente"); dpg.add_combo([], tag="c_i_mat", width=200)
            dpg.add_button(label="PROCESSAR", width=120, height=50, callback=trigger_process)

    dpg.add_separator()
    with dpg.group():
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play/Pause", callback=lambda: setattr(state, 'playing', not state.playing))
            dpg.add_slider_int(label="Speed", default_value=5, max_value=50, width=150, callback=lambda s,a: setattr(state, 'speed', a))
            dpg.add_spacer(width=20)
            dpg.add_text("T: 0.000s", tag="txt_info")
            dpg.add_text("Pt: 0/0", tag="txt_sample")
        dpg.add_slider_int(label="", tag="timeline", width=-1, default_value=0, max_value=1000, callback=cb_timeline)

    with dpg.tab_bar():
        
        with dpg.tab(label="Monitoramento"):
            with dpg.plot(height=200, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Tempo", tag="ax_rms_x")
                with dpg.plot_axis(dpg.mvYAxis, label="Tensão (V)", tag="ax_rms_yv"): pass
                dpg.add_drag_line(tag="cur_rms", vertical=True)
            
            with dpg.plot(height=200, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Tempo", tag="ax_rms_x_i")
                with dpg.plot_axis(dpg.mvYAxis, label="Corrente (A)", tag="ax_rms_yi"): pass
                dpg.add_drag_line(tag="cur_rms_i", vertical=True)

            with dpg.plot(height=200, width=-1, no_mouse_pos=True):
                dpg.add_plot_legend()
                x_ax = dpg.add_plot_axis(dpg.mvXAxis, label="Componentes", no_gridlines=True, no_tick_marks=True)
                dpg.set_axis_ticks(x_ax, (("Zero", 1.1), ("Pos", 2.1), ("Neg", 3.1)))
                dpg.set_axis_limits(x_ax, 0.5, 4.0)
                with dpg.plot_axis(dpg.mvYAxis, label="Magnitude", tag="ax_seq_y"):
                    dpg.add_bar_series([1, 2, 3], [0, 0, 0], tag="bar_seq", weight=0.25, label="Atual")
                    dpg.add_bar_series([1.25, 2.25, 3.25], [0, 0, 0], tag="bar_seq_ref", weight=0.25, label="Ref")

        with dpg.tab(label="Análise Vetorial"):
            with dpg.group(horizontal=True):
                with dpg.plot(label="Clarke XY", width=600, height=350):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Alpha")
                    with dpg.plot_axis(dpg.mvYAxis, label="Beta", tag="ax_clk_xy_y"):
                        dpg.add_line_series([], [], label="Rastro", tag="tr_clk")
                        dpg.add_scatter_series([], [], label="Pt", tag="dot_clk")
                
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_combo(CURVE_LIST, default_value=CURVE_LIST[0], width=200, callback=cb_tcc, tag="in_cv")
                        dpg.add_input_float(label="Pk", default_value=25.0, width=70, callback=cb_tcc, tag="in_pk")
                        dpg.add_input_float(label="Dl", default_value=0.5, width=70, callback=cb_tcc, tag="in_td")
                    
                    with dpg.plot(label="Curva TCC (Zoom Livre)", width=-1, height=350):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="I (A)", log_scale=True)
                        y_ax = dpg.add_plot_axis(dpg.mvYAxis, label="t (s)", log_scale=True)
                        # Sem limits fixos para permitir zoom livre
                        
                        dpg.add_line_series([], [], label="Curva", tag="ser_tcc", parent=y_ax)
                        
                        # Atuais (Círculos)
                        with dpg.theme(tag="th_curr"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                        
                        dpg.add_scatter_series([], [], label="A", tag="dot_tcc_a", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_curr")
                        dpg.add_scatter_series([], [], label="B", tag="dot_tcc_b", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_curr")
                        dpg.add_scatter_series([], [], label="C", tag="dot_tcc_c", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_curr")
                        
                        # Ref (Cruzes)
                        with dpg.theme(tag="th_ref"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Cross, category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)

                        dpg.add_scatter_series([], [], label="A(Ref)", tag="dot_tcc_a_ref", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_ref")
                        dpg.add_scatter_series([], [], label="B(Ref)", tag="dot_tcc_b_ref", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_ref")
                        dpg.add_scatter_series([], [], label="C(Ref)", tag="dot_tcc_c_ref", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_ref")

            with dpg.plot(label="Clarke no Tempo", width=-1, height=250):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Tempo", tag="ax_clk_t")
                with dpg.plot_axis(dpg.mvYAxis, label="Valor", tag="ax_clk_time_y"): pass
                dpg.add_drag_line(tag="cur_clk_t", vertical=True)

dpg.create_viewport(title='Suite V21 (Legends & Zoom)', width=1280, height=800)
dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window("Win", True)
while dpg.is_dearpygui_running(): update_visuals(); dpg.render_dearpygui_frame()
dpg.destroy_context()