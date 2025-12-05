import streamlit as st
import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Análise Harmônica - IEEE 34", layout="wide")

st.title("📊 Analisador de Harmônicas (Exclusão da Fundamental)")
st.markdown("""
Esta ferramenta carrega os dados `.mat`, identifica sinais trifásicos e foca na 
**Distorção Harmônica**, permitindo ignorar a frequência fundamental (60Hz).
""")

# --- Barra Lateral: Upload ---
st.sidebar.header("Carregar Dados")
uploaded_files = st.sidebar.file_uploader("Carregue os arquivos .mat", type=["mat"], accept_multiple_files=True)

def calcular_fft(time, signal):
    """Calcula a FFT unilateral e retorna frequências e magnitudes."""
    dt = np.mean(np.diff(time))
    fs = 1 / dt
    n = len(signal)
    
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(n, dt)
    
    pos_mask = fft_freq >= 0
    freqs = fft_freq[pos_mask]
    magnitude = 2.0/n * np.abs(fft_vals[pos_mask])
    
    return freqs, magnitude, fs

if uploaded_files:
    file_map = {f.name: f for f in uploaded_files}
    selected_file_name = st.selectbox("Selecione o arquivo:", list(file_map.keys()))
    
    if selected_file_name:
        file_obj = file_map[selected_file_name]
        
        try:
            mat_data = sio.loadmat(file_obj, squeeze_me=True, struct_as_record=False)
            
            if 'ts' in mat_data:
                ts_struct = mat_data['ts']
                available_vars = [attr for attr in dir(ts_struct) if attr.startswith('ts_')]
                
                if not available_vars:
                    st.error("Nenhuma variável 'ts_' encontrada.")
                else:
                    # 1. Seleção do Sinal
                    col_sel1, col_sel2 = st.columns(2)
                    with col_sel1:
                        selected_var = st.selectbox("Selecione o Sinal:", available_vars)
                    
                    # Extração segura
                    signal_obj = getattr(ts_struct, selected_var)
                    try:
                        t = signal_obj.Time
                        raw_data = signal_obj.Data
                    except AttributeError:
                        t = signal_obj.time
                        raw_data = signal_obj.signals.values

                    # 2. Tratamento Trifásico (Seleção de Fase)
                    if raw_data.ndim > 1:
                        num_phases = raw_data.shape[1]
                        with col_sel2:
                            phase_idx = st.selectbox(
                                "O sinal possui múltiplas fases. Escolha uma:", 
                                range(num_phases), 
                                format_func=lambda x: f"Fase {x+1} (Coluna {x})"
                            )
                        y = raw_data[:, phase_idx]
                    else:
                        y = raw_data
                        with col_sel2:
                            st.info("Sinal Monofásico identificado.")

                    # 3. Análise FFT e Filtros
                    freqs, mag, fs = calcular_fft(t, y)
                    
                    # FILTRO PRINCIPAL: Remover Fundamental e DC
                    # Mantém apenas frequências >= 90Hz (segunda harmônica é 120Hz)
                    mask_harmonics = freqs >= 90 
                    
                    freqs_h = freqs[mask_harmonics]
                    mag_h = mag[mask_harmonics]

                    # --- Visualização ---
                    st.divider()
                    col_graph, col_data = st.columns([2, 1])
                    
                    with col_graph:
                        st.subheader("Espectro Harmônico (A partir da 2ª Ordem)")
                        
                        # Slider de Zoom
                        max_freq = st.slider("Frequência Máxima (Hz)", 120, 2000, 1000)
                        
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=freqs_h, 
                            y=mag_h, 
                            name='Harmônicas',
                            marker_color='red'
                        ))
                        
                        fig.update_layout(
                            xaxis_title="Frequência (Hz)",
                            yaxis_title="Magnitude",
                            xaxis_range=[90, max_freq], # Começa em 90 para cortar o ruído perto da fundamental
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col_data:
                        st.subheader("Tabela de Harmônicas")
                        st.caption("Picos identificados em múltiplos de 60Hz")
                        
                        # Gerar Tabela Automática
                        harmonics_data = []
                        fundamental = 60
                        
                        # Busca magnitudes nas frequências exatas (2ª até 20ª harmônica)
                        for order in range(2, 21): 
                            target_f = fundamental * order
                            # Encontra o índice mais próximo no vetor de frequências da FFT
                            idx_closest = (np.abs(freqs - target_f)).argmin()
                            
                            measured_f = freqs[idx_closest]
                            val = mag[idx_closest]
                            
                            # Filtro visual para tabela: só mostra se tiver relevância (> 0.001)
                            if val > 0.001: 
                                harmonics_data.append({
                                    "Ordem": f"{order}ª",
                                    "Freq (Hz)": f"{measured_f:.1f}",
                                    "Magnitude": f"{val:.4f}"
                                })
                        
                        if harmonics_data:
                            st.dataframe(pd.DataFrame(harmonics_data), hide_index=True)
                        else:
                            st.info("Nenhuma harmônica significativa encontrada.")

            else:
                st.error("Estrutura 'ts' não encontrada no arquivo.")
        except Exception as e:
            st.error(f"Erro: {e}")