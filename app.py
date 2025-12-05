import streamlit as st
import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Análise Harmônica Trifásica", layout="wide")

st.title("📊 Análise Harmônica Trifásica (até 10kHz)")
st.markdown("""
Esta ferramenta analisa a distorção harmônica ignorando a fundamental (60Hz).
Os dados são separados automaticamente por fase (A, B, C) e o espectro vai até 10.000 Hz.
""")

# --- Barra Lateral: Upload ---
st.sidebar.header("Carregar Dados")
uploaded_files = st.sidebar.file_uploader("Carregue os arquivos .mat", type=["mat"], accept_multiple_files=True)

def calcular_fft(time, signal):
    """Calcula FFT e retorna frequências, magnitudes e taxa de amostragem."""
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    n = len(signal)
    
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, dt)
    
    # Lado positivo apenas
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    magnitude = 2.0/n * np.abs(fft_vals[pos_mask])
    
    return freqs, magnitude, fs

def renderizar_aba_fase(t, y, nome_fase, cor_grafico):
    """Função auxiliar para gerar o conteúdo de cada aba (Fase A, B ou C)."""
    
    # 1. Cálculo da FFT
    freqs, mag, fs = calcular_fft(t, y)
    nyquist = fs / 2  # Limite teórico (10kHz para fs=20kHz)

    # 2. Filtro: Remover Fundamental e DC (> 90Hz)
    mask_harmonics = freqs >= 90
    freqs_h = freqs[mask_harmonics]
    mag_h = mag[mask_harmonics]

    # --- Layout da Aba ---
    col_graf, col_dados = st.columns([3, 1])

    with col_graf:
        st.subheader(f"Espectro de Frequência - {nome_fase}")
        
        # Slider de Zoom (até Nyquist)
        max_view = st.slider(f"Frequência Máxima ({nome_fase})", 
                             min_value=500, 
                             max_value=int(nyquist), 
                             value=2000, # Valor inicial amigável, mas pode ir até 10k
                             step=100)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=freqs_h, 
            y=mag_h, 
            name=f'Harmônicas {nome_fase}',
            marker_color=cor_grafico
        ))
        
        fig.update_layout(
            xaxis_title="Frequência (Hz)",
            yaxis_title="Magnitude",
            xaxis_range=[90, max_view], # Começa em 90Hz
            height=450,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_dados:
        st.subheader(f"Tabela ({nome_fase})")
        st.caption("Maiores magnitudes (Múltiplos de 60Hz)")

        # Gerar Tabela de Harmônicas (2ª até 50ª ordem ou limite de Nyquist)
        harmonics_data = []
        fundamental = 60
        
        for order in range(2, 168): # 167 * 60 ~= 10020 Hz
            target_f = fundamental * order
            
            if target_f > nyquist:
                break
                
            # Encontrar pico mais próximo
            idx_closest = (np.abs(freqs - target_f)).argmin()
            measured_f = freqs[idx_closest]
            val = mag[idx_closest]
            
            # Só mostrar na tabela se tiver relevância mínima para não poluir
            if val > 1e-4: 
                harmonics_data.append({
                    "Ordem": f"{order}ª",
                    "Freq (Hz)": f"{measured_f:.0f}",
                    "Mag": val  # Mantém float para ordenação correta se precisar
                })
        
        if harmonics_data:
            df_h = pd.DataFrame(harmonics_data)
            # Formatação para exibição
            st.dataframe(
                df_h.style.format({"Mag": "{:.4f}"}), 
                use_container_width=True,
                height=400
            )
        else:
            st.info("Sem harmônicas significativas.")

# --- Lógica Principal ---
if uploaded_files:
    file_map = {f.name: f for f in uploaded_files}
    selected_file = st.selectbox("Arquivo:", list(file_map.keys()))
    
    if selected_file:
        try:
            mat = sio.loadmat(file_map[selected_file], squeeze_me=True, struct_as_record=False)
            
            if 'ts' in mat:
                ts = mat['ts']
                vars_list = [v for v in dir(ts) if v.startswith('ts_')]
                
                sel_var = st.selectbox("Sinal para Análise:", vars_list)
                signal_obj = getattr(ts, sel_var)
                
                # Extração de Tempo e Dados
                try:
                    t = signal_obj.Time
                    y_raw = signal_obj.Data
                except:
                    t = signal_obj.time
                    y_raw = signal_obj.signals.values

                # --- Verificação de Fases ---
                # Se for array 2D com 3 colunas, assume Trifásico
                if y_raw.ndim > 1 and y_raw.shape[1] == 3:
                    st.success(f"Sinal Trifásico Identificado (Fs = {1/np.mean(np.diff(t)):.0f} Hz)")
                    
                    # Criação das Abas
                    tab_a, tab_b, tab_c = st.tabs(["⚡ Fase A", "⚡ Fase B", "⚡ Fase C"])
                    
                    with tab_a:
                        renderizar_aba_fase(t, y_raw[:, 0], "Fase A", "red")
                    with tab_b:
                        renderizar_aba_fase(t, y_raw[:, 1], "Fase B", "blue")
                    with tab_c:
                        renderizar_aba_fase(t, y_raw[:, 2], "Fase C", "green")
                        
                elif y_raw.ndim > 1 and y_raw.shape[1] == 2:
                     # Caso raro de 2 fases/sinais
                    tab1, tab2 = st.tabs(["Canal 1", "Canal 2"])
                    with tab1: renderizar_aba_fase(t, y_raw[:, 0], "Canal 1", "orange")
                    with tab2: renderizar_aba_fase(t, y_raw[:, 1], "Canal 2", "cyan")
                else:
                    # Monofásico
                    st.info("Sinal Monofásico")
                    if y_raw.ndim > 1: y_raw = y_raw.flatten()
                    renderizar_aba_fase(t, y_raw, "Sinal Único", "purple")

            else:
                st.error("Estrutura 'ts' não encontrada no arquivo.")
                
        except Exception as e:
            st.error(f"Erro ao ler arquivo: {e}")