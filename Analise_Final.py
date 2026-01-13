# %%
# ==============================================================================
# 1. IMPORTS E CONFIGURAÇÕES GERAIS
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import scipy.io as sio
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import warnings
import matplotlib as mpl

# Suprimir avisos
warnings.filterwarnings('ignore')

# --- CONSTANTES DO SISTEMA ---
# Define o instante exato da falta para plotagem e cálculos
INSTANTE_FALTA = 0.5 / 3  # ~0.1667 segundos

# --- CONFIGURAÇÕES DE ESTILO (ABNT/IEEE) ---
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'lines.linewidth': 2.0,
    'axes.grid': True,
    'grid.alpha': 0.3
})

# Paleta de Cores Consistente
CORES = {
    'A': '#0066CC', 'B': '#CC0000', 'C': '#009900',
    'MRN': '#1f77b4', 'T2F': '#ff7f0e',
    'ComReg': '#2ca02c', 'SemReg': '#d62728',
    'ComTerra': '#9467bd', 'SemTerra': '#8c564b'
}

print("✅ Ambiente Configurado. Instante da falta definido em: {:.4f} s".format(INSTANTE_FALTA))


# ==============================================================================
# 2. MOTOR DE PROCESSAMENTO E CÁLCULOS
# ==============================================================================

class ProcessadorAvancado:
    def __init__(self, t, V, I, freq=60):
        self.t = np.array(t).flatten()
        self.v = self._garantir_3_fases(V)
        self.i = self._garantir_3_fases(I)
        self.freq = freq

        # Taxa de amostragem
        if len(self.t) > 1:
            self.dt = float(self.t[1] - self.t[0])
            self.fs = 1.0 / self.dt
        else:
            self.fs = 60 * 256
            self.dt = 1 / self.fs

        self.samples_per_cycle = int(self.fs / self.freq)

    def _garantir_3_fases(self, arr):
        arr = np.array(arr)
        if arr.ndim == 1: return np.column_stack([arr, arr, arr])
        if arr.shape[0] == 3 and arr.shape[1] > 3: return arr.T
        return arr

    def calcular_rms(self, sinal):
        """RMS móvel (janela de 1 ciclo)."""
        window = max(1, self.samples_per_cycle)
        return np.sqrt(uniform_filter1d(sinal ** 2, size=window, axis=0))

    def get_phasors(self, t_janela_inicio, n_cycles=1):
        """Retorna fasores complexos (Mag, Angle) para um instante."""
        idx = np.searchsorted(self.t, t_janela_inicio)
        window = int(self.samples_per_cycle * n_cycles)
        phasors = []
        for fase in range(3):
            if idx + window > len(self.v):
                seg = self.v[idx:, fase]
            else:
                seg = self.v[idx:idx + window, fase]

            # FFT simples para pegar fundamental
            if len(seg) == 0:
                phasors.append(0j)
                continue

            fft_res = np.fft.rfft(seg)
            freqs = np.fft.rfftfreq(len(seg), 1 / self.fs)
            idx_60 = np.argmin(np.abs(freqs - 60))
            val = fft_res[idx_60] if idx_60 < len(fft_res) else 0
            # Normalizar magnitude pela janela (aprox)
            mag = np.abs(val) * 2 / len(seg)
            angle = np.angle(val)
            phasors.append(mag * np.exp(1j * angle))
        return np.array(phasors)

    def get_sym_components(self, t_instante):
        """Retorna magnitudes [Zero, Pos, Neg]."""
        phasors = self.get_phasors(t_instante)
        a = np.exp(1j * 2 * np.pi / 3)
        A_mat = np.array([[1, 1, 1], [1, a ** 2, a], [1, a, a ** 2]]) / 3
        seq = A_mat @ phasors
        return np.abs(seq)  # [V0, V1, V2]

    def get_fft_metrics(self, t_inicio, t_fim):
        """Calcula V1, V3, THD e retorna espectro para plot."""
        idx_ini = np.searchsorted(self.t, t_inicio)
        idx_fim = np.searchsorted(self.t, t_fim)
        seg = self.v[idx_ini:idx_fim, 0]  # Fase A por padrão

        if len(seg) == 0: return None

        # Janela Hanning
        w = np.hanning(len(seg))
        X = np.fft.rfft(seg * w)
        freqs = np.fft.rfftfreq(len(seg), d=self.dt)
        mag = (2.0 / np.sum(w)) * np.abs(X)

        # Métricas
        idx_60 = np.argmin(np.abs(freqs - 60))
        idx_180 = np.argmin(np.abs(freqs - 180))

        v1 = mag[idx_60] if idx_60 < len(mag) else 0
        v3 = mag[idx_180] if idx_180 < len(mag) else 0

        harmonics_sum = np.sum(mag[idx_60 + 1:] ** 2)
        thd = (np.sqrt(harmonics_sum) / v1) * 100 if v1 > 0 else 0

        return {'freqs': freqs, 'mag': mag, 'V1': v1, 'V3': v3, 'THD': thd}


# ==============================================================================
# 3. LEITURA E ESTRUTURAÇÃO DE DADOS (O "CENARIO")
# ==============================================================================

def parse_nome_arquivo(nome):
    nome = nome.lower()
    topo = 'T2F' if 't2f' in nome else 'MRN'
    reg = 'sem_regulador' if ('_sr_' in nome or 'sem_reg' in nome or 'semreg' in nome) else 'com_regulador'
    terra = 'sem_aterramento' if ('sem_terra' in nome or 'semterra' in nome) else 'com_aterramento'

    # Tipo Falta (Dedução básica - ajuste conforme seus nomes reais)
    if 'abc' in nome:
        falta = 'ABC'
    elif 'a-g' in nome or 'ag' in nome or 'a_terra' in nome:
        falta = 'A_terra'
    elif 'ab' in nome:
        falta = 'AB'
    elif 'bc' in nome:
        falta = 'BC'
    else:
        falta = 'pleno_funcionamento'  # Default se não achar falta

    # Barra da Falta (Assume-se que o nome do arquivo indica onde foi a falta)
    barra_falta = '820'  # Default
    if '816' in nome:
        barra_falta = '816'
    elif '822' in nome:
        barra_falta = '822'

    return topo, reg, terra, falta, barra_falta


def carregar_dados_brutos(caminho):
    try:
        t, V_dict, I_dict = None, {}, {}
        # Tenta HDF5
        with h5py.File(caminho, 'r') as f:
            for k in f.keys():
                if 't' == k or 'time' in k: t = f[k][()].flatten()
                if 'V_' in k and 'raw' in k: V_dict[k] = f[k][()].T
                if 'I_' in k and 'raw' in k: I_dict[k] = f[k][()].T
    except:
        # Tenta MAT
        mat = sio.loadmat(caminho, squeeze_me=False)
        for k in mat.keys():
            if 't' == k or 'time' in k: t = mat[k].flatten()
            if k.startswith('V_') and 'raw' in k: V_dict[k] = mat[k]
            if k.startswith('I_') and 'raw' in k: I_dict[k] = mat[k]

    return t, V_dict, I_dict


def processar_arquivo_cenario(caminho):
    topo, reg, terra, falta, b_falta = parse_nome_arquivo(caminho.name)
    t, V_dict, I_dict = carregar_dados_brutos(caminho)

    if t is None: return None

    # Estrutura do Cenário conforme especificação
    cenario = {
        "topologia": topo, "regulador": reg, "aterramento": terra,
        "tipo_falta": falta, "barra_falta": b_falta, "resultados_por_barra": {}
    }

    # Processa cada barra encontrada no arquivo
    barras_interesse = ['800', '816', '820', '822']
    for b in barras_interesse:
        # Busca chaves correspondentes (insensível a maiúsculas/minúsculas se necessário)
        key_v = next((k for k in V_dict if f"_{b}_" in k), None)
        key_i = next((k for k in I_dict if f"_{b}_" in k), None)

        if key_v and key_i:
            proc = ProcessadorAvancado(t, V_dict[key_v], I_dict[key_i])

            # --- CÁLCULOS PRÉVIOS (CACHE) ---
            # 1. RMS
            v_rms = proc.calcular_rms(proc.v)
            i_rms = proc.calcular_rms(proc.i)

            # 2. Métricas de Falta (USANDO A CONSTANTE DE TEMPO)
            t_falta_inicio = INSTANTE_FALTA
            t_falta_fim = INSTANTE_FALTA + 0.1  # Janela de 100ms

            # Simétricas na falta (pequeno delay para estabilizar após transitório)
            v_sym = proc.get_sym_components(t_falta_inicio + 0.02)
            i_sym = proc.get_sym_components(t_falta_inicio + 0.02)

            # FFT na falta
            fft_data = proc.get_fft_metrics(t_falta_inicio, t_falta_fim)

            # Índices
            idx_falta = np.searchsorted(proc.t, t_falta_inicio)
            # Pega o máximo num intervalo após o início da falta
            janela_max = int(0.1 * proc.fs)
            if len(i_rms) > idx_falta + janela_max:
                i_falta_max = np.max(i_rms[idx_falta:idx_falta + janela_max, :])
            else:
                i_falta_max = 0

            v2_v1 = (v_sym[2] / v_sym[1]) * 100 if v_sym[1] > 0 else 0
            v3_v1 = (fft_data['V3'] / fft_data['V1']) * 100 if fft_data and fft_data['V1'] > 0 else 0
            thd = fft_data['THD'] if fft_data else 0

            cenario["resultados_por_barra"][b] = {
                "t": proc.t, "v_rms": v_rms, "i_rms": i_rms,
                "V0_V1_V2": v_sym, "I0_I1_I2": i_sym,
                "fft_dados": fft_data,
                "indices": {
                    "I_falta_max": i_falta_max, "V2_V1_percent": v2_v1,
                    "V3_V1_percent": v3_v1, "THD_V_percent": thd,
                    "V1_pu": fft_data['V1'] if fft_data else 0,
                    "V3_pu": fft_data['V3'] if fft_data else 0
                }
            }

    return cenario


# ==============================================================================
# 4. FUNÇÕES GERADORAS (BLOCOS 1 a 5)
# ==============================================================================

# --- BLOCO 1: RMS (Tensão 820/822 e Corrente 800) ---
def gerar_bloco_1(lista_cenarios, pasta_saida):
    print("🔹 Gerando Bloco 1: Gráficos RMS...")
    subpasta = Path(pasta_saida) / "Bloco1_RMS"
    subpasta.mkdir(parents=True, exist_ok=True)

    # Agrupar para plotagem
    cenarios_foco = [c for c in lista_cenarios if c['tipo_falta'] == 'A_terra']

    for c in cenarios_foco:
        nome_base = f"{c['topologia']}_{c['regulador']}_{c['aterramento']}_{c['tipo_falta']}"

        # 1.1 Tensão RMS 820
        if '820' in c['resultados_por_barra']:
            dados = c['resultados_por_barra']['820']
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dados['t'], dados['v_rms'], label=['Va', 'Vb', 'Vc'])

            # LINHA VERTICAL DA FALTA
            ax.axvline(x=INSTANTE_FALTA, color='k', linestyle='--', linewidth=1.5, label='Início Falta')

            ax.set_title(f"Tensão RMS 820 - {c['topologia']} {c['regulador']} {c['aterramento']}")
            ax.set_ylabel("Tensão (V)")
            ax.set_xlabel("Tempo (s)")
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(subpasta / f"TensaoRMS_barra820_{nome_base}.png")
            plt.close()

        # 1.2 Corrente RMS 800
        if '800' in c['resultados_por_barra']:
            dados = c['resultados_por_barra']['800']
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(dados['t'], dados['i_rms'], label=['Ia', 'Ib', 'Ic'])

            # LINHA VERTICAL DA FALTA
            ax.axvline(x=INSTANTE_FALTA, color='k', linestyle='--', linewidth=1.5, label='Início Falta')

            ax.set_title(f"Corrente RMS 800 - {c['topologia']} {c['regulador']} {c['aterramento']}")
            ax.set_ylabel("Corrente (A)")
            ax.set_xlabel("Tempo (s)")
            plt.legend(loc='best')
            plt.tight_layout()
            plt.savefig(subpasta / f"CorrenteRMS_barra800_{nome_base}.png")
            plt.close()


# --- BLOCO 2: 3ª HARMÔNICA (FFT) ---
def gerar_bloco_2(lista_cenarios, pasta_saida):
    print("🔹 Gerando Bloco 2: FFT e Tabela Harmônica...")
    subpasta = Path(pasta_saida) / "Bloco2_FFT"
    subpasta.mkdir(parents=True, exist_ok=True)

    tabela_dados = []

    # Foco na barra 820
    for c in lista_cenarios:
        if '820' not in c['resultados_por_barra']: continue

        res_barra = c['resultados_por_barra']['820']
        fft = res_barra['fft_dados']
        idx = res_barra['indices']

        if fft:
            # Gráfico
            fig, ax = plt.subplots(figsize=(10, 5))
            mask = fft['freqs'] <= 500
            ax.stem(fft['freqs'][mask], fft['mag'][mask], basefmt=" ", markerfmt="ko", linefmt="k-")

            # Destaques
            for f_target in [60, 180]:
                ax.axvline(x=f_target, color='r', linestyle='--', alpha=0.5)
                # Coloca texto apenas se magnitude for relevante para não poluir
                if np.max(fft['mag']) > 0:
                    y_pos = np.max(fft['mag']) * 0.9
                    ax.text(f_target, y_pos, f"{f_target}Hz", color='r', ha='center', backgroundcolor='white')

            ax.set_title(f"FFT Tensão 820 - {c['topologia']} {c['regulador']} {c['aterramento']}")
            nome_fig = f"FFT_V_barra820_{c['topologia']}_{c['regulador']}_{c['aterramento']}_{c['tipo_falta']}.png"
            plt.savefig(subpasta / nome_fig)
            plt.close()

            # Dados Tabela
            tabela_dados.append({
                "topologia": c['topologia'], "regulador": c['regulador'],
                "aterramento": c['aterramento'], "tipo_falta": c['tipo_falta'],
                "condicao": "falta", "V1_pu": idx['V1_pu'], "V3_pu": idx['V3_pu'],
                "V3_V1_percent": idx['V3_V1_percent'], "THD_V_percent": idx['THD_V_percent']
            })

    # Salva CSV Consolidado
    if tabela_dados:
        df = pd.DataFrame(tabela_dados)
        df.to_csv(subpasta / "Resumo_FFT_tensao_barra820.csv", index=False)


# --- BLOCO 3: SIMÉTRICAS ---
def gerar_bloco_3(lista_cenarios, pasta_saida):
    print("🔹 Gerando Bloco 3: Componentes Simétricas...")
    subpasta = Path(pasta_saida) / "Bloco3_Simetricas"
    subpasta.mkdir(parents=True, exist_ok=True)

    dados_simetricas = []

    # Foco barra 820
    for c in lista_cenarios:
        if '820' not in c['resultados_por_barra']: continue
        res = c['resultados_por_barra']['820']

        v_sym = res['V0_V1_V2']
        dados_simetricas.append({
            "id_cenario": f"{c['topologia']}_{c['regulador'][:3]}_{c['aterramento'][:3]}",  # Label curto
            "V0": v_sym[0], "V1": v_sym[1], "V2": v_sym[2],
            "tipo_falta": c['tipo_falta']
        })

    df = pd.DataFrame(dados_simetricas)
    if df.empty: return

    # Gráficos por tipo de falta
    for falta in df['tipo_falta'].unique():
        df_falta = df[df['tipo_falta'] == falta]
        if df_falta.empty: continue

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(df_falta))
        w = 0.25

        ax.bar(x - w, df_falta['V0'], w, label='V0 (Zero)', color='gray')
        ax.bar(x, df_falta['V1'], w, label='V1 (Pos)', color='blue')
        ax.bar(x + w, df_falta['V2'], w, label='V2 (Neg)', color='red')

        ax.set_xticks(x)
        ax.set_xticklabels(df_falta['id_cenario'], rotation=45, ha='right')
        ax.set_title(f"Componentes Simétricas (Tensão) - Falta {falta}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(subpasta / f"Simetricas_V_barra820_{falta}.png")
        plt.close()


# --- BLOCO 4: COMPARAÇÃO T2F vs MRN ---
def gerar_bloco_4(lista_cenarios, pasta_saida):
    print("🔹 Gerando Bloco 4: Comparativo T2F vs MRN...")
    subpasta = Path(pasta_saida) / "Bloco4_Comparativo"
    subpasta.mkdir(parents=True, exist_ok=True)

    # Filtro: Com Regulador e Com Aterramento (conforme spec)
    filtro = lambda c: c['regulador'] == 'com_regulador' and c['aterramento'] == 'com_aterramento'
    cenarios_validos = [c for c in lista_cenarios if filtro(c)]

    comparativo = []
    for c in cenarios_validos:
        if '820' in c['resultados_por_barra'] and '822' in c['resultados_por_barra']:
            comparativo.append({
                "topologia": c['topologia'],
                "tipo_falta": c['tipo_falta'],
                "I_falta_820": c['resultados_por_barra']['820']['indices']['I_falta_max'],
                "V2_V1_822": c['resultados_por_barra']['822']['indices']['V2_V1_percent']
            })

    df = pd.DataFrame(comparativo)
    if df.empty: return

    # Gráfico 4.1: Corrente de Falta (Agrupado por Tipo Falta)
    pivot_i = df.pivot(index='tipo_falta', columns='topologia', values='I_falta_820')
    if not pivot_i.empty:
        pivot_i.plot(kind='bar', figsize=(10, 6), color=[CORES['MRN'], CORES['T2F']])
        plt.title("Corrente Máxima de Falta (820) - MRN vs T2F")
        plt.ylabel("Corrente (A)")
        plt.tight_layout()
        plt.savefig(subpasta / "Ifalta_barra820_MRN_vs_T2F.png")
        plt.close()

    # Gráfico 4.2: Desequilíbrio (Agrupado por Tipo Falta)
    pivot_v = df.pivot(index='tipo_falta', columns='topologia', values='V2_V1_822')
    if not pivot_v.empty:
        pivot_v.plot(kind='bar', figsize=(10, 6), color=[CORES['MRN'], CORES['T2F']])
        plt.title("Desequilíbrio V2/V1 (822) - MRN vs T2F")
        plt.ylabel("Desequilíbrio (%)")
        plt.tight_layout()
        plt.savefig(subpasta / "DesequilibrioV_barra822_MRN_vs_T2F.png")
        plt.close()


# --- BLOCO 5: PAPEL DO REGULADOR ---
def gerar_bloco_5(lista_cenarios, pasta_saida):
    print("🔹 Gerando Bloco 5: Perfil de Tensão e Regulador...")
    subpasta = Path(pasta_saida) / "Bloco5_Regulador"
    subpasta.mkdir(parents=True, exist_ok=True)

    # 5.1 Perfil em Pleno Funcionamento
    cenarios_pleno = [c for c in lista_cenarios if c['tipo_falta'] == 'pleno_funcionamento']

    if cenarios_pleno:
        fig, ax = plt.subplots(figsize=(10, 6))
        barras = ['800', '816', '820', '822']
        x = range(len(barras))

        for c in cenarios_pleno:
            vals = []
            for b in barras:
                if b in c['resultados_por_barra']:
                    # Média RMS (assumindo regime permanente estável)
                    v_med = np.mean(c['resultados_por_barra'][b]['v_rms'])
                    vals.append(v_med)
                else:
                    vals.append(None)

            label = f"{c['topologia']} {c['regulador']}"
            ls = '-' if c['regulador'] == 'com_regulador' else '--'
            color = CORES['MRN'] if c['topologia'] == 'MRN' else CORES['T2F']

            if None not in vals:
                ax.plot(x, vals, marker='o', label=label, linestyle=ls, color=color)

        ax.set_xticks(x);
        ax.set_xticklabels(barras)
        ax.set_title("Perfil de Tensão - Pleno Funcionamento")
        ax.set_ylabel("Tensão Média (V)")
        plt.legend()
        plt.savefig(subpasta / "PerfilTensao_plenoFunc.png")
        plt.close()


# ==============================================================================
# 5. EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    # 🔧 CONFIGURAÇÃO DO CAMINHO
    PASTA_MAT = r"C:/Users/Leonardo Felipe/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/"
    PASTA_RESULTADOS = "./Resultados_Secao4_Analise_Tese"
    Path(PASTA_RESULTADOS).mkdir(parents=True, exist_ok=True)

    arquivos = list(Path(PASTA_MAT).glob("*.mat"))
    print(f"🚀 Iniciando processamento de {len(arquivos)} cenários...")

    # 1. Carregar e Processar Tudo
    lista_cenarios = []
    for arq in arquivos:
        try:
            cenario = processar_arquivo_cenario(arq)
            if cenario:
                lista_cenarios.append(cenario)
                print(f"   -> Processado: {cenario['topologia']} {cenario['regulador']} {cenario['tipo_falta']}")
        except Exception as e:
            print(f"   ❌ Erro em {arq.name}: {e}")

    print(f"\n✅ {len(lista_cenarios)} cenários carregados. Gerando Blocos de Análise...\n")

    # 2. Gerar Blocos
    if lista_cenarios:
        gerar_bloco_1(lista_cenarios, PASTA_RESULTADOS)
        gerar_bloco_2(lista_cenarios, PASTA_RESULTADOS)
        gerar_bloco_3(lista_cenarios, PASTA_RESULTADOS)
        gerar_bloco_4(lista_cenarios, PASTA_RESULTADOS)
        gerar_bloco_5(lista_cenarios, PASTA_RESULTADOS)

        # 3. Exportar Tabela Mestra (Resumo Geral)
        tabela_mestra = []
        for c in lista_cenarios:
            if '820' in c['resultados_por_barra']:
                idx = c['resultados_por_barra']['820']['indices']
                tabela_mestra.append({
                    "topologia": c['topologia'], "regulador": c['regulador'],
                    "aterramento": c['aterramento'], "tipo_falta": c['tipo_falta'],
                    "barra": "820",
                    "I_falta_max": idx['I_falta_max'], "V2_V1": idx['V2_V1_percent'],
                    "V3_V1": idx['V3_V1_percent'], "THD": idx['THD_V_percent']
                })
        pd.DataFrame(tabela_mestra).to_csv(Path(PASTA_RESULTADOS) / "Resumo_Geral_Cenarios.csv", index=False)
        print("\n🏆 Processamento Completo! Verifique a pasta 'Resultados_Secao4_Analise_Tese_v2'.")


if __name__ == "__main__":
    main()