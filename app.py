import streamlit as st
import scipy.io as sio
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# Configuração da página para ocupar toda a largura (Wide Mode)
st.set_page_config(page_title="Comparador de Harmônicas IEEE 34", layout="wide")

st.title("📊 Comparador de Faltas e Harmônicas (Multi-Arquivo)")
st.markdown("""
Esta ferramenta compara a **Distorção Harmônica** entre múltiplos arquivos `.mat`.
Os gráficos superiores mostram a sobreposição de todos os arquivos para cada Fase (A, B, C).
""")

# --- Função de Cálculo FFT ---
def calcular_fft(time, signal):
    """Retorna frequências, magnitudes e taxa de amostragem."""
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

# --- Barra Lateral: Configurações ---
st.sidebar.header("1. Carregar Dados")
uploaded_files = st.sidebar.file_uploader(
    "Selecione os arquivos .mat para comparar", 
    type=["mat"], 
    accept_multiple_files=True
)

# Configurações de Visualização
st.sidebar.header("2. Configuração do Gráfico")
max_freq_view = st.sidebar.slider("Frequência Máxima (Zoom)", 120, 10000, 2000, step=100)
min_mag_view = st.sidebar.number_input("Magnitude Mínima (Filtro Visual)", value=0.001, format="%.4f")

if uploaded_files:
    # --- Passo 1: Leitura e Identificação de Variáveis Comuns ---
    data_cache = {}
    common_vars = None
    
    for uploaded_file in uploaded_files:
        try:
            # Lê o arquivo
            mat = sio.loadmat(uploaded_file, squeeze_me=True, struct_as_record=False)
            if 'ts' in mat:
                ts = mat['ts']
                # Lista variáveis deste arquivo (ex: ts_I_800)
                vars_this_file = set([v for v in dir(ts) if v.startswith('ts_')])
                
                # Armazena no cache
                data_cache[uploaded_file.name] = {'ts': ts, 'vars': vars_this_file}
                
                # Atualiza interseção de variáveis (para garantir que selecionamos algo que existe em todos)
                if common_vars is None:
                    common_vars = vars_this_file
                else:
                    common_vars = common_vars.intersection(vars_this_file)
            else:
                st.warning(f"Arquivo ignorado (sem struct 'ts'): {uploaded_file.name}")
        except Exception as e:
            st.error(f"Erro ao ler {uploaded_file.name}: {e}")

    if common_vars:
        # Seletor de Variável
        sorted_vars = sorted(list(common_vars))
        selected_var = st.selectbox("Selecione o Sinal para Comparar (Comum a todos os arquivos):", sorted_vars)
        
        # --- Passo 2: Processamento dos Dados para o Gráfico ---
        # Estrutura para plotagem: { 'Fase A': [ (nome_arq, x, y), ... ], 'Fase B': ... }
        plot_data = {'Fase A': [], 'Fase B': [], 'Fase C': []}
        
        for fname, content in data_cache.items():
            ts_struct = content['ts']
            signal_obj = getattr(ts_struct, selected_var)
            
            # Extração Time/Data
            try:
                t = signal_obj.Time
                y_raw = signal_obj.Data
            except:
                t = signal_obj.time
                y_raw = signal_obj.signals.values
            
            # Cálculo FFT para cada fase
            # Verifica se é trifásico (N, 3) ou monofásico (N,)
            if y_raw.ndim > 1 and y_raw.shape[1] >= 3:
                fases = [('Fase A', 0), ('Fase B', 1), ('Fase C', 2)]
            elif y_raw.ndim > 1 and y_raw.shape[1] == 2:
                 fases = [('Fase A', 0), ('Fase B', 1)] # Assume A e B
            else:
                fases = [('Fase A', None)] # Monofásico trata como A

            for fase_name, idx in fases:
                # Seleciona dados da fase
                y_sig = y_raw[:, idx] if idx is not None else y_raw.flatten()
                
                # FFT
                freqs, mags, fs = calcular_fft(t, y_sig)
                
                # Filtro: >= 90Hz (Remove Fundamental e DC)
                mask = freqs >= 90
                
                # Armazena para plotagem
                plot_data[fase_name].append({
                    'file': fname,
                    'freqs': freqs[mask],
                    'mags': mags[mask]
                })

        # --- Passo 3: Visualização "COMPARATIVO DIRETO" (Todas as fases na mesma tela) ---
        st.divider()
        st.subheader(f"📈 Comparativo: {selected_var}")
        
        # Layout de Colunas para Fases
        cols = st.columns(3)
        phase_titles = ['Fase A', 'Fase B', 'Fase C']
        
        for i, col in enumerate(cols):
            phase = phase_titles[i]
            with col:
                st.markdown(f"**{phase}**")
                
                fig = go.Figure()
                
                # Adiciona um traço para cada arquivo nesta fase
                for dataset in plot_data.get(phase, []):
                    # Filtra visualização pelo Slider
                    mask_view = dataset['freqs'] <= max_freq_view
                    x_view = dataset['freqs'][mask_view]
                    y_view = dataset['mags'][mask_view]
                    
                    # Filtra magnitude para limpar gráfico (opcional)
                    mask_noise = y_view > min_mag_view
                    
                    fig.add_trace(go.Bar(
                        x=x_view[mask_noise],
                        y=y_view[mask_noise],
                        name=dataset['file'], # Nome do arquivo na legenda
                        opacity=0.8
                    ))
                
                fig.update_layout(
                    xaxis_title="Freq (Hz)",
                    yaxis_title="Mag",
                    xaxis_range=[90, max_freq_view],
                    legend=dict(orientation="h", y=-0.2), # Legenda em baixo
                    margin=dict(l=20, r=20, t=20, b=20),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

        # --- Passo 4: Seção "POR ARQUIVO" (Detalhamento) ---
        st.divider()
        st.subheader("📂 Detalhamento por Arquivo")
        
        for fname in data_cache.keys():
            with st.expander(f"Detalhes: {fname}", expanded=False):
                # Recupera os dados calculados anteriormente para este arquivo
                # Filtra plot_data para pegar apenas este arquivo nas 3 fases
                
                col_d1, col_d2, col_d3 = st.columns(3)
                cols_detalhe = [col_d1, col_d2, col_d3]
                
                for i, phase in enumerate(phase_titles):
                    # Encontra os dados deste arquivo e fase específica
                    dados_fase = next((d for d in plot_data[phase] if d['file'] == fname), None)
                    
                    if dados_fase:
                        with cols_detalhe[i]:
                            # Tabela de Picos (Top 10 harmônicas)
                            df_picos = pd.DataFrame({
                                'Freq (Hz)': dados_fase['freqs'],
                                'Mag': dados_fase['mags']
                            })
                            # Pega apenas picos relevantes (> min_mag_view)
                            df_picos = df_picos[df_picos['Mag'] > min_mag_view]
                            # Ordena por magnitude
                            df_picos = df_picos.sort_values(by='Mag', ascending=False).head(10)
                            
                            st.markdown(f"**{phase} - Top Harmônicas**")
                            if not df_picos.empty:
                                st.dataframe(
                                    df_picos.style.format("{:.4f}"), 
                                    hide_index=True,
                                    use_container_width=True
                                )
                            else:
                                st.info("Sem picos significativos.")

    else:
        st.info("Nenhuma variável comum encontrada nos arquivos selecionados ou nenhum arquivo carregado.")
else:
    st.info("Utilize o menu lateral para carregar os arquivos .mat.")