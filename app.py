import streamlit as st
import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import re

# Configuração visual da página
st.set_page_config(page_title="Análise Visual - Corrente Primeiro", layout="wide")

st.title("📊 Análise Visual de Harmônicas (I & V)")
st.markdown("""
Comparação visual com foco na **Corrente** seguida pela **Tensão**.
Indicadores de **THD** e Tabelas de Harmônicas incluídos.
""")

# --- Funções Auxiliares ---
def calcular_fft(time, signal):
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, dt)
    
    # Apenas frequências positivas
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    mags = 2.0/n * np.abs(fft_vals[pos_mask])
    
    return freqs, mags, fs

def calcular_thd(freqs, mags):
    """Calcula THD (%) considerando fundamental em ~60Hz."""
    idx_60 = (np.abs(freqs - 60)).argmin()
    mag_fund = mags[idx_60]
    
    if mag_fund == 0: return 0
    
    harm_sq_sum = 0
    for h in range(2, 51):
        target = 60 * h
        if target > freqs[-1]: break
        idx = (np.abs(freqs - target)).argmin()
        harm_sq_sum += mags[idx]**2
        
    thd = (np.sqrt(harm_sq_sum) / mag_fund) * 100
    return thd

def ordenar_pontos(lista_encontrada):
    """Ordena conforme a preferência do usuário."""
    ordem_desejada = ['800', 'T2F', '818', '820', '822']
    
    def get_sort_key(ponto):
        if ponto in ordem_desejada:
            return ordem_desejada.index(ponto)
        return 999 

    return sorted(lista_encontrada, key=get_sort_key)

# --- Componente de Plotagem Visual ---
def plotar_analise_visual(titulo, dados_cache, var_name, max_freq, min_mag):
    st.markdown(f"## ⚡ {titulo}")
    
    fases_cfg = [('Fase A', 0, 'red'), ('Fase B', 1, 'blue'), ('Fase C', 2, 'green')]
    dados_heatmap = []

    # Loop principal das fases
    for fase_nome, idx_fase, cor_base in fases_cfg:
        
        thd_results = {}
        plot_traces = []
        
        for fname, content in dados_cache.items():
            if hasattr(content['ts'], var_name):
                signal_obj = getattr(content['ts'], var_name)
                try:
                    t = signal_obj.Time
                    y_raw = signal_obj.Data
                except:
                    t = signal_obj.time
                    y_raw = signal_obj.signals.values
                
                # Seleção Fase/Sinal
                if y_raw.ndim > 1 and y_raw.shape[1] > idx_fase:
                    y_sig = y_raw[:, idx_fase]
                else:
                    if idx_fase > 0: continue
                    y_sig = y_raw.flatten()

                # Processamento
                freqs, mags, _ = calcular_fft(t, y_sig)
                thd = calcular_thd(freqs, mags)
                thd_results[fname] = thd
                
                # Dados para Tabela
                for h in range(2, 10): # 2ª até 9ª
                    f_h = 60 * h
                    idx_h = (np.abs(freqs - f_h)).argmin()
                    val_h = mags[idx_h]
                    dados_heatmap.append({
                        'Arquivo': fname,
                        'Fase': fase_nome,
                        'Harmônica': f"{h}ª ({f_h}Hz)",
                        'Magnitude': val_h
                    })

                # Filtro visual
                mask = (freqs >= 90) & (freqs <= max_freq)
                mask_noise = mags[mask] > min_mag
                
                if np.any(mask_noise):
                    plot_traces.append(go.Bar(
                        x=freqs[mask][mask_noise],
                        y=mags[mask][mask_noise],
                        name=f"{fname} (THD: {thd:.1f}%)",
                        opacity=0.75
                    ))

        # Exibição Visual (Métricas + Gráfico)
        st.markdown(f"#### {fase_nome}")
        
        if thd_results:
            cols_metrics = st.columns(len(thd_results))
            for i, (arq, val_thd) in enumerate(thd_results.items()):
                delta_color = "normal"
                if val_thd > 5.0: delta_color = "inverse"
                cols_metrics[i].metric(
                    label=f"THD - {arq}", 
                    value=f"{val_thd:.2f}%", 
                    delta="Alto Risco" if val_thd > 8 else None,
                    delta_color=delta_color
                )

        fig = go.Figure(data=plot_traces)
        fig.update_layout(
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Frequência (Hz)",
            yaxis_title="Magnitude",
            legend=dict(orientation="h", y=1.1, x=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.divider()

    # Tabela Global Comparativa (Sem gradiente de cor para evitar erros)
    if dados_heatmap:
        st.markdown("### 📋 Tabela Comparativa de Harmônicas")
        df_heat = pd.DataFrame(dados_heatmap)
        
        tab_h1, tab_h2, tab_h3 = st.tabs(["Fase A", "Fase B", "Fase C"])
        
        for i, (tab, f_nome) in enumerate(zip([tab_h1, tab_h2, tab_h3], ['Fase A', 'Fase B', 'Fase C'])):
            with tab:
                df_fase = df_heat[df_heat['Fase'] == f_nome]
                if not df_fase.empty:
                    pivot = df_fase.pivot_table(
                        index='Harmônica', 
                        columns='Arquivo', 
                        values='Magnitude'
                    )
                    # CORREÇÃO: Usando formatação simples para evitar erro de Matplotlib
                    st.dataframe(
                        pivot.style.format("{:.4f}"),
                        use_container_width=True
                    )
                else:
                    st.info("Sem dados para esta fase.")

# --- Barra Lateral e Carga ---
st.sidebar.header("1. Upload de Arquivos")
uploaded_files = st.sidebar.file_uploader("Arquivos .mat", type=["mat"], accept_multiple_files=True)

st.sidebar.header("2. Ajustes Visuais")
max_freq_view = st.sidebar.slider("Zoom Frequência (Hz)", 120, 5000, 1200)
min_mag_view = st.sidebar.number_input("Filtro Magnitude Mínima", 0.0, 1.0, 0.001, format="%.4f")

if uploaded_files:
    data_cache = {}
    all_vars = set()
    
    for f in uploaded_files:
        try:
            mat = sio.loadmat(f, squeeze_me=True, struct_as_record=False)
            if 'ts' in mat:
                data_cache[f.name] = {'ts': mat['ts']}
                all_vars.update([v for v in dir(mat['ts']) if v.startswith('ts_')])
        except: pass

    pontos_set = set()
    for v in all_vars:
        match = re.search(r'ts_[VI]_(.+)', v)
        if match: pontos_set.add(match.group(1))
    
    lista_ordenada = ordenar_pontos(list(pontos_set))

    # Criação das Abas Principais
    st.write("")
    tabs = st.tabs([f"📍 {p}" for p in lista_ordenada])

    for i, ponto in enumerate(lista_ordenada):
        with tabs[i]:
            
            # --- ORDEM INVERTIDA: PRIMEIRO CORRENTE (I), DEPOIS TENSÃO (V) ---
            
            # 1. Corrente
            var_i = f"ts_I_{ponto}"
            tem_i = any(hasattr(d['ts'], var_i) for d in data_cache.values())
            
            if tem_i:
                plotar_analise_visual(f"Corrente {ponto}", data_cache, var_i, max_freq_view, min_mag_view)
            
            # Separador se houver os dois
            if tem_i:
                st.markdown("---") 

            # 2. Tensão
            var_v = f"ts_V_{ponto}"
            tem_v = any(hasattr(d['ts'], var_v) for d in data_cache.values())
            
            if tem_v:
                plotar_analise_visual(f"Tensão {ponto}", data_cache, var_v, max_freq_view, min_mag_view)

else:
    st.info("Por favor, carregue os arquivos .mat na barra lateral.")