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
        if L < 10: raise ValueError("Dados insuficientes.")
        
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
        
        self.v_phasors = self._calc_phasors(self.v_raw)
        self.i_phasors = self._calc_phasors(self.i_raw)
        
        self.i_seq = np.nan_to_num(self._seq_from_phasors(self.i_phasors))

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

    def _calc_phasors(self, x):
        rot = np.exp(-1j * 2 * np.pi * self.freq * self.t)
        ph = np.zeros_like(x, dtype=complex)
        for k in range(3):
            ph[:,k] = scipy.ndimage.uniform_filter1d(x[:,k] * rot, self.samples) * np.sqrt(2)
        return np.nan_to_num(ph)

    def _seq_from_phasors(self, ph):
        a = np.exp(1j * 2*np.pi/3)
        v0 = (ph[:,0] + ph[:,1] + ph[:,2])/3.0
        v1 = (ph[:,0] + a*ph[:,1] + a**2*ph[:,2])/3.0
        v2 = (ph[:,0] + a**2*ph[:,1] + a*ph[:,2])/3.0
        return np.stack([np.abs(v0), np.abs(v1), np.abs(v2)], axis=1)

class AppState:
    def __init__(self):
        self.proc_curr = None
        self.proc_ref = None
        self.loaded_raw = None
        self.loaded_ref = None 
        
        self.idx = 0      
        self.idx_ref = 0  
        self.sync_ref = True
        
        self.playing = False
        self.speed = 5
        self.pickup = 25.0; self.dial = 0.5; self.curve_idx = 0

state = AppState()

CURVES = {
    "IEC Standard": (0.14, 0.0, 0.02), "IEC Very": (13.5, 0.0, 1.0),
    "IEC Extreme": (80.0, 0.0, 2.0), "IEEE Mod": (0.0515, 0.1140, 0.02),
    "IEEE Very": (19.61, 0.4910, 2.0), "IEEE Extreme": (28.2, 0.1217, 2.0)
}
CURVE_LIST = list(CURVES.keys())

C_A = (0, 255, 255, 255); C_B = (255, 50, 50, 255); C_C = (0, 255, 0, 255)
C_REF = (150, 150, 150, 100)
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
            dpg.set_value("status", "Principal Carregado.")
            
        elif user_data == "ref":
            state.loaded_ref = raw
            dpg.configure_item("c_v_ref", items=keys)
            dpg.configure_item("c_i_ref", items=keys)
            dpg.set_value("c_v_ref", v_guess)
            dpg.set_value("c_i_ref", i_guess)
            dpg.set_value("status", f"Ref Carregada: {os.path.basename(path)}")
            
    except: dpg.set_value("status", "Erro Load")

def trigger_process():
    try:
        # 1. PROCESSA PRINCIPAL
        if state.loaded_raw:
            v_key = dpg.get_value("c_v_mat")
            i_key = dpg.get_value("c_i_mat")
            raw = state.loaded_raw
            t_key = next((k for k in raw.keys() if k.lower() in ['t', 'time']), None)
            
            if t_key:
                proc = SignalProcessor(raw[t_key], raw[v_key], raw[i_key])
                state.proc_curr = proc; state.idx = 0
                dpg.configure_item("timeline", max_value=len(proc.t)-1)
                setup_plots(proc, "curr")

        # 2. PROCESSA REFERÊNCIA (SE EXISTIR)
        if state.loaded_ref:
            v_key_ref = dpg.get_value("c_v_ref")
            i_key_ref = dpg.get_value("c_i_ref")
            raw_ref = state.loaded_ref
            t_key_ref = next((k for k in raw_ref.keys() if k.lower() in ['t', 'time']), None)
            
            if t_key_ref:
                proc_ref = SignalProcessor(raw_ref[t_key_ref], raw_ref[v_key_ref], raw_ref[i_key_ref])
                state.proc_ref = proc_ref
                dpg.configure_item("timeline_ref", max_value=len(proc_ref.t)-1)
                setup_plots(proc_ref, "ref")

        dpg.set_value("status", "Processamento Concluído!")
        
    except Exception as e:
        dpg.set_value("status", f"Erro: {str(e)}")

# ==============================================================================
# 3. LÓGICA & PLOTS
# ==============================================================================
def cb_timeline(sender, app_data):
    state.idx = int(app_data); state.playing = False
    if state.sync_ref and state.proc_ref:
        state.idx_ref = min(state.idx, len(state.proc_ref.t)-1)
        dpg.set_value("timeline_ref", state.idx_ref)
    update_visuals()

def cb_timeline_ref(sender, app_data):
    if not state.sync_ref:
        state.idx_ref = int(app_data); update_visuals()

def cb_sync(sender, app_data):
    state.sync_ref = app_data
    if state.sync_ref and state.proc_ref:
        state.idx_ref = min(state.idx, len(state.proc_ref.t)-1)
        dpg.set_value("timeline_ref", state.idx_ref)
        update_visuals()

def setup_plots(p, target):
    suf = "" if target == "curr" else "_ref"
    if target == "curr":
        colors = [C_A, C_B, C_C]; thick = 2.0
    else:
        colors = [list(c[:3])+[100] for c in [C_A, C_B, C_C]]; thick = 1.0

    t = p.t.tolist()
    
    def line(tag_base, y_arr, col_idx, label_name):
        tag = tag_base + suf
        col = colors[col_idx]
        if "rms_v" in tag_base: parent = "ax_rms_yv"
        elif "rms_i" in tag_base: parent = "ax_rms_yi"
        elif "clk_time" in tag_base: parent = "ax_clk_time_y"
        elif "xy_full" in tag_base: parent = "ax_clk_xy_y"
        else: return

        if len(y_arr) != len(t): return
        lbl = label_name + (" (Ref)" if target == "ref" else "")

        if dpg.does_item_exist(tag): dpg.set_value(tag, [t, y_arr.tolist()])
        else:
            if dpg.does_item_exist(parent):
                if "xy_full" in tag_base:
                    dpg.add_line_series(p.i_clarke['alpha'].tolist(), p.i_clarke['beta'].tolist(), parent=parent, tag=tag, label=lbl)
                else:
                    dpg.add_line_series(t, y_arr.tolist(), parent=parent, tag=tag, label=lbl)
                with dpg.theme() as tm:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
                        dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, thick, category=dpg.mvThemeCat_Plots)
                dpg.bind_item_theme(tag, tm)

    def init_arrow(tag_base, parent_ax, col_idx, label):
        tag = tag_base + suf
        col = colors[col_idx]
        lbl = label + (" (Ref)" if target == "ref" else "")
        if not dpg.does_item_exist(tag):
            dpg.add_line_series([0,0], [0,0], parent=parent_ax, tag=tag, label=lbl)
            with dpg.theme() as tm:
                with dpg.theme_component(dpg.mvLineSeries):
                    dpg.add_theme_color(dpg.mvPlotCol_Line, col, category=dpg.mvThemeCat_Plots)
                    dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, thick+1, category=dpg.mvThemeCat_Plots) 
            dpg.bind_item_theme(tag, tm)

    try:
        line("rms_va", p.v_rms[:,0], 0, "Va"); line("rms_vb", p.v_rms[:,1], 1, "Vb"); line("rms_vc", p.v_rms[:,2], 2, "Vc")
        line("rms_ia", p.i_rms[:,0], 0, "Ia"); line("rms_ib", p.i_rms[:,1], 1, "Ib"); line("rms_ic", p.i_rms[:,2], 2, "Ic")
        line("clk_time_a", p.i_clarke['alpha'], 0, "Alpha"); line("clk_time_b", p.i_clarke['beta'], 1, "Beta")
        if target == "ref": line("xy_full", p.i_clarke['beta'], 0, "Rastro Ref") 

        init_arrow("ph_va", "ax_ph_v_y", 0, "Va"); init_arrow("ph_vb", "ax_ph_v_y", 1, "Vb"); init_arrow("ph_vc", "ax_ph_v_y", 2, "Vc")
        init_arrow("ph_ia", "ax_ph_i_y", 0, "Ia"); init_arrow("ph_ib", "ax_ph_i_y", 1, "Ib"); init_arrow("ph_ic", "ax_ph_i_y", 2, "Ic")

        if target == "curr":
            for ax in ["ax_rms_x", "ax_rms_yv", "ax_rms_x_i", "ax_rms_yi", "ax_clk_t", "ax_clk_time_y"]: dpg.fit_axis_data(ax)
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
            if state.sync_ref and state.proc_ref:
                state.idx_ref = min(state.idx, len(state.proc_ref.t)-1)
                dpg.set_value("timeline_ref", state.idx_ref)
        
        idx = int(state.idx); idx_ref = int(state.idx_ref)
        t = state.proc_curr.t[idx]
        dpg.set_value("txt_info", f"T(P): {t:.3f}s")
        
        for c in ["cur_rms", "cur_rms_i", "cur_clk_t"]: 
            if dpg.does_item_exist(c): dpg.set_value(c, t)
            
        seq_vals = state.proc_curr.i_seq[idx].tolist()
        dpg.set_value("bar_seq", [[1, 2, 3], seq_vals])
        
        a = state.proc_curr.i_clarke['alpha']; b = state.proc_curr.i_clarke['beta']
        s = max(0, idx-300)
        dpg.set_value("tr_clk", [a[s:idx].tolist(), b[s:idx].tolist()])
        dpg.set_value("dot_clk", [[a[idx]], [b[idx]]])
        
        i_inst = state.proc_curr.i_rms[idx]
        trips = [calc_trip(v) for v in i_inst]
        dpg.set_value("dot_tcc_a", [[i_inst[0]], [trips[0]]]); dpg.set_value("dot_tcc_b", [[i_inst[1]], [trips[1]]]); dpg.set_value("dot_tcc_c", [[i_inst[2]], [trips[2]]])

        def upd_ph(proc, suf, ix):
            v_c = proc.v_phasors[ix]; i_c = proc.i_phasors[ix]
            dpg.set_value(f"ph_va{suf}", [[0, v_c[0].real], [0, v_c[0].imag]])
            dpg.set_value(f"ph_vb{suf}", [[0, v_c[1].real], [0, v_c[1].imag]])
            dpg.set_value(f"ph_vc{suf}", [[0, v_c[2].real], [0, v_c[2].imag]])
            dpg.set_value(f"ph_ia{suf}", [[0, i_c[0].real], [0, i_c[0].imag]])
            dpg.set_value(f"ph_ib{suf}", [[0, i_c[1].real], [0, i_c[1].imag]])
            dpg.set_value(f"ph_ic{suf}", [[0, i_c[2].real], [0, i_c[2].imag]])
        upd_ph(state.proc_curr, "", idx)

        if state.proc_ref and idx_ref < len(state.proc_ref.t):
            t_ref = state.proc_ref.t[idx_ref]
            dpg.set_value("txt_info_ref", f"T(R): {t_ref:.3f}s")
            for c in ["cur_rms_ref", "cur_rms_i_ref", "cur_clk_t_ref"]:
                if dpg.does_item_exist(c): dpg.configure_item(c, show=True); dpg.set_value(c, t_ref)
            dpg.set_value("bar_seq_ref", [[1.25, 2.25, 3.25], state.proc_ref.i_seq[idx_ref].tolist()])
            i_r = state.proc_ref.i_rms[idx_ref]; tr_r = [calc_trip(v) for v in i_r]
            dpg.set_value("dot_tcc_a_ref", [[i_r[0]], [tr_r[0]]]); dpg.set_value("dot_tcc_b_ref", [[i_r[1]], [tr_r[1]]]); dpg.set_value("dot_tcc_c_ref", [[i_r[2]], [tr_r[2]]])
            upd_ph(state.proc_ref, "_ref", idx_ref)
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
    x = np.logspace(0, 4, 200); y = [calc_trip(v) for v in x]
    dpg.set_value("ser_tcc", [x.tolist(), y])

def cb_tcc(s, a):
    if s == "in_pk": state.pickup = a
    if s == "in_td": state.dial = a
    if s == "in_cv": state.curve_idx = CURVE_LIST.index(a)
    update_tcc()

# ==============================================================================
# GUI
# ==============================================================================
with dpg.file_dialog(directory_selector=False, show=False, callback=load_file_cb, id="fd_curr", width=700, height=500, user_data="curr"):
    dpg.add_file_extension(".mat", color=(0, 255, 0))
with dpg.file_dialog(directory_selector=False, show=False, callback=load_file_cb, id="fd_ref", width=700, height=500, user_data="ref"):
    dpg.add_file_extension(".mat", color=(255, 255, 0))

dpg.set_global_font_scale(1)

with dpg.window(tag="Win"):
    with dpg.collapsing_header(label="Configuração de Arquivos e Canais", default_open=True):
        # GRID LAYOUT CORRIGIDO (SEM SEPARATOR VERTICAL)
        with dpg.group(horizontal=True):
            
            # --- COLUNA 1: PRINCIPAL ---
            with dpg.group(width=400):
                dpg.add_text("DADOS PRINCIPAIS", color=(0, 255, 255))
                dpg.add_button(label="1. Carregar Principal", callback=lambda: dpg.show_item("fd_curr"), height=40, width=200)
                dpg.add_spacer(height=5)
                dpg.add_text("Canal Tensão (V):")
                dpg.add_combo([], tag="c_v_mat", width=350)
                dpg.add_text("Canal Corrente (I):")
                dpg.add_combo([], tag="c_i_mat", width=350)

            # ESPAÇADOR LARGO PARA SEPARAR
            dpg.add_spacer(width=50)

            # --- COLUNA 2: REFERÊNCIA ---
            with dpg.group(width=400):
                dpg.add_text("DADOS REFERÊNCIA (OPCIONAL)", color=(255, 100, 100))
                dpg.add_button(label="2. Carregar Ref", callback=lambda: dpg.show_item("fd_ref"), height=40, width=200)
                dpg.add_spacer(height=5)
                dpg.add_text("Ref Tensão (V):")
                dpg.add_combo([], tag="c_v_ref", width=350)
                dpg.add_text("Ref Corrente (I):")
                dpg.add_combo([], tag="c_i_ref", width=350)

        dpg.add_spacer(height=10)
        dpg.add_separator()
        dpg.add_button(label="PROCESSAR TUDO", width=-1, height=60, callback=trigger_process)
        dpg.add_text("Status...", tag="status")

    dpg.add_separator()
    
    # --- CONTROLES DE TEMPO ---
    with dpg.group():
        with dpg.group(horizontal=True):
            dpg.add_button(label="Play/Pause", callback=lambda: setattr(state, 'playing', not state.playing), height=40, width=100)
            dpg.add_slider_int(label="Speed", default_value=5, max_value=50, width=150, callback=lambda s,a: setattr(state, 'speed', a))
            dpg.add_spacer(width=30)
            dpg.add_checkbox(label="Sync Ref", tag="chk_sync", default_value=True, callback=cb_sync)
            dpg.add_spacer(width=20)
            dpg.add_text("T(P): 0.00s", tag="txt_info")
            dpg.add_text("", tag="txt_info_ref", color=(255,100,100))

        dpg.add_text("Tempo Principal")
        dpg.add_slider_int(label="", tag="timeline", width=-1, height=20, default_value=0, max_value=1000, callback=cb_timeline)
        
        dpg.add_text("Tempo Referência")
        with dpg.theme(tag="th_red"):
            with dpg.theme_component(dpg.mvSliderInt):
                dpg.add_theme_color(dpg.mvThemeCol_SliderGrab, (200, 50, 50), category=dpg.mvThemeCat_Core)
        dpg.add_slider_int(label="", tag="timeline_ref", width=-1, height=20, default_value=0, max_value=1000, callback=cb_timeline_ref)
        dpg.bind_item_theme("timeline_ref", "th_red")

    with dpg.tab_bar():
        with dpg.tab(label="Monitoramento"):
            with dpg.plot(height=250, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Tempo", tag="ax_rms_x")
                with dpg.plot_axis(dpg.mvYAxis, label="V", tag="ax_rms_yv"): pass
                dpg.add_drag_line(tag="cur_rms", vertical=True)
                dpg.add_drag_line(tag="cur_rms_ref", vertical=True, color=(200,50,50,200), show=False)
            
            with dpg.plot(height=250, width=-1):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Tempo", tag="ax_rms_x_i")
                with dpg.plot_axis(dpg.mvYAxis, label="A", tag="ax_rms_yi"): pass
                dpg.add_drag_line(tag="cur_rms_i", vertical=True)
                dpg.add_drag_line(tag="cur_rms_i_ref", vertical=True, color=(200,50,50,200), show=False)

            with dpg.plot(height=250, width=-1, no_mouse_pos=True):
                dpg.add_plot_legend()
                x_ax = dpg.add_plot_axis(dpg.mvXAxis, label="Comp", no_gridlines=True, no_tick_marks=True)
                dpg.set_axis_ticks(x_ax, (("Zero", 1.1), ("Pos", 2.1), ("Neg", 3.1)))
                dpg.set_axis_limits(x_ax, 0.5, 4.0)
                with dpg.plot_axis(dpg.mvYAxis, label="Mag", tag="ax_seq_y"):
                    dpg.add_bar_series([1, 2, 3], [0, 0, 0], tag="bar_seq", weight=0.25, label="Atual")
                    dpg.add_bar_series([1.25, 2.25, 3.25], [0, 0, 0], tag="bar_seq_ref", weight=0.25, label="Ref")

        with dpg.tab(label="Análise Vetorial"):
            with dpg.group(horizontal=True):
                with dpg.plot(label="Clarke XY", width=650, height=400):
                    dpg.add_plot_axis(dpg.mvXAxis, label="Alpha")
                    with dpg.plot_axis(dpg.mvYAxis, label="Beta", tag="ax_clk_xy_y"):
                        dpg.add_line_series([], [], label="Rastro", tag="tr_clk")
                        dpg.add_scatter_series([], [], label="Pt", tag="dot_clk")
                
                with dpg.group():
                    with dpg.group(horizontal=True):
                        dpg.add_combo(CURVE_LIST, default_value=CURVE_LIST[0], width=200, callback=cb_tcc, tag="in_cv")
                        dpg.add_input_float(label="Pk", default_value=25.0, width=100, callback=cb_tcc, tag="in_pk")
                        dpg.add_input_float(label="Dl", default_value=0.5, width=100, callback=cb_tcc, tag="in_td")
                    
                    with dpg.plot(label="Curva TCC", width=-1, height=400):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="I (A)", log_scale=True)
                        y_ax = dpg.add_plot_axis(dpg.mvYAxis, label="t (s)", log_scale=True)
                        dpg.add_line_series([], [], label="Curva", tag="ser_tcc", parent=y_ax)
                        
                        with dpg.theme(tag="th_curr"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Circle, category=dpg.mvThemeCat_Plots)
                        dpg.add_scatter_series([], [], label="A", tag="dot_tcc_a", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_curr")
                        dpg.add_scatter_series([], [], label="B", tag="dot_tcc_b", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_curr")
                        dpg.add_scatter_series([], [], label="C", tag="dot_tcc_c", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_curr")
                        
                        with dpg.theme(tag="th_ref"):
                            with dpg.theme_component(dpg.mvScatterSeries):
                                dpg.add_theme_style(dpg.mvPlotStyleVar_Marker, dpg.mvPlotMarker_Cross, category=dpg.mvThemeCat_Plots)
                                dpg.add_theme_style(dpg.mvPlotStyleVar_MarkerSize, 6, category=dpg.mvThemeCat_Plots)
                        dpg.add_scatter_series([], [], label="A(Ref)", tag="dot_tcc_a_ref", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_ref")
                        dpg.add_scatter_series([], [], label="B(Ref)", tag="dot_tcc_b_ref", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_ref")
                        dpg.add_scatter_series([], [], label="C(Ref)", tag="dot_tcc_c_ref", parent=y_ax); dpg.bind_item_theme(dpg.last_item(), "th_ref")

            with dpg.plot(label="Clarke Tempo", width=-1, height=300):
                dpg.add_plot_legend()
                dpg.add_plot_axis(dpg.mvXAxis, label="Tempo", tag="ax_clk_t")
                with dpg.plot_axis(dpg.mvYAxis, label="Valor", tag="ax_clk_time_y"): pass
                dpg.add_drag_line(tag="cur_clk_t", vertical=True)
                dpg.add_drag_line(tag="cur_clk_t_ref", vertical=True, color=(200,50,50,200), show=False)

            with dpg.group(horizontal=True):
                with dpg.plot(label="Fasores Tensão", width=650, height=350):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Re")
                    with dpg.plot_axis(dpg.mvYAxis, label="Im", tag="ax_ph_v_y"): pass
                with dpg.plot(label="Fasores Corrente", width=-1, height=350):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="Re")
                    with dpg.plot_axis(dpg.mvYAxis, label="Im", tag="ax_ph_i_y"): pass

dpg.create_viewport(title='Suite V28 (Stable Final)', width=1280, height=900)
dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window("Win", True)
while dpg.is_dearpygui_running(): update_visuals(); dpg.render_dearpygui_frame()
dpg.destroy_context()