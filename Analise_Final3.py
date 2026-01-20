# ======================================================================
# SCRIPT FINAL - NOMES PERSONALIZADOS (ANTES/DEPOIS DO TRAFO)
# ======================================================================

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import csv
from math import isfinite

# ----------------------------------------------------------------------
# 1. CONFIGURAÇÕES GERAIS E MAPA DE NOMES
# ----------------------------------------------------------------------

FREQUENCIA_REDE = 60.0  # Hz
TEMPO_FALTA = 0.5 / 3.0  # Instante da falta (s)

# --- AQUI ESTÁ A SUBSTITUIÇÃO DOS NOMES ---
MAPA_LEGENDAS = {
    "I_T2F_raw": "Antes do Trafo Isolador",
    "I_T2F1_raw": "Depois do Trafo Isolador",
    "V_T2F_raw": "Tensão Antes do Trafo",
    "V_T2F1_raw": "Tensão Depois do Trafo"
}

def cm_to_inch(value):
    return value / 2.54

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 10,
    "figure.figsize": (cm_to_inch(16), cm_to_inch(10)),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
    "legend.framealpha": 1.0,
    "legend.edgecolor": "black"
})

PASTA_RAIZ_RESULTADOS = Path("Resultados_Tese_Cap5_Comparativo_Final")
PASTA_RAIZ_RESULTADOS.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 2. LISTA DE ARQUIVOS (GRUPOS)
# ----------------------------------------------------------------------
lista_arquivos_cap5 = [
    (
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__R_822_-_Falta_AB_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__R_822_-_Falta_AB_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__R_822_-_Falta_AB_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__R_822_-_Falta_AB_py.mat',
    ),
    (
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__R_822_-_Falta_ABC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__R_822_-_Falta_ABC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__R_822_-_Falta_ABC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__R_822_-_Falta_ABC_py.mat',
    ),
    (
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__R_822_-_Falta_AC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__R_822_-_Falta_AC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__R_822_-_Falta_AC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__R_822_-_Falta_AC_py.mat',
    ),
    (
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__R_822_-_Falta_BC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__R_822_-_Falta_BC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__R_822_-_Falta_BC_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__R_822_-_Falta_BC_py.mat',
    ),
    (
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao__Sem_Falta_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_sem_terra__Sem_Falta_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR__Sem_Falta_py.mat',
        r'C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/Qualificacao_SR_sem_terra__Sem_Falta_py.mat',
    ),
]

RELIGADORES_PARAMS = {
    "R_Montante_T2F": {"Pickup": 20, "TMS_TD": 0.3},
    "R_Jusante_T2F1": {"Pickup": 35, "TMS_TD": 0.3}
}

CURVAS_DEFS = {
    "IEC_C1": (0.14, 0.0, 0.02),
    "IEC_C2": (13.5, 0.0, 1.0),
    "IEC_C3": (80.0, 0.0, 2.0),
    "IEEE_U1": (0.0515, 0.114, 0.02),
    "IEEE_U2": (19.61, 0.491, 2.0),
    "IEEE_U3": (28.2, 0.1217, 2.0)
}


# ----------------------------------------------------------------------
# 3. FUNÇÕES AUXILIARES
# ----------------------------------------------------------------------

def identifying_group_by_name(nome_arquivo):
    if "Falta_ABC" in nome_arquivo:
        return "Falta ABC"
    elif "Falta_AB" in nome_arquivo:
        return "Falta AB"
    elif "Falta_AC" in nome_arquivo:
        return "Falta AC"
    elif "Falta_BC" in nome_arquivo:
        return "Falta BC"
    elif "Sem_Falta" in nome_arquivo:
        return "Sem Falta"
    else:
        return "Desconhecido"


def formatar_nome_caso(nome_arquivo):
    is_sr = "SR" in nome_arquivo
    is_st = "sem_terra" in nome_arquivo
    if is_sr and is_st:
        return "Sem Regulador e Sem Terra"
    elif is_sr and not is_st:
        return "Sem Regulador"
    elif not is_sr and is_st:
        return "Sem Terra"
    else:
        return "Com Regulador e Com Terra"


def calcular_rms_movel(sinal, tempo, freq_rede=FREQUENCIA_REDE):
    if len(tempo) < 2: return np.zeros_like(sinal)
    dt = tempo[1] - tempo[0]
    if dt <= 0: return np.zeros_like(sinal)
    fs = 1.0 / dt
    janela = int(round(fs / freq_rede))
    if janela < 1: janela = 1
    sinal_quadrado = sinal ** 2
    janela_media = np.ones(janela) / janela
    rms = np.sqrt(np.convolve(sinal_quadrado, janela_media, mode="same"))
    return rms


def get_imax_envelope_vetor(tempo, dados_tres_fases):
    rms_a = calcular_rms_movel(dados_tres_fases[:, 0], tempo)
    rms_b = calcular_rms_movel(dados_tres_fases[:, 1], tempo)
    rms_c = calcular_rms_movel(dados_tres_fases[:, 2], tempo)
    return np.maximum.reduce([rms_a, rms_b, rms_c])


def clarke_power_invariant(ia, ib, ic):
    k = np.sqrt(2.0 / 3.0)
    i_alpha = k * (ia - 0.5 * ib - 0.5 * ic)
    i_beta = k * ((np.sqrt(3) / 2) * ib - (np.sqrt(3) / 2) * ic)
    return i_alpha, i_beta


def padronizar_dados(f_handle, var_name):
    if var_name not in f_handle: return None
    d = np.array(f_handle[var_name])
    if d.ndim == 1:
        d = d.reshape(-1, 1)
    elif d.ndim == 2 and d.shape[0] < d.shape[1]:
        d = d.T
    linhas, col = d.shape
    d_out = np.zeros((linhas, 3))
    if col == 1:
        d_out[:, 0] = d[:, 0]
    elif col == 2:
        d_out[:, 0] = d[:, 0]
        d_out[:, 1] = d[:, 1]
    else:
        d_out[:, 0:3] = d[:, 0:3]
    return d_out


def calcular_tempo_curva(M, TMS_TD, curva_nome):
    if M <= 1.0: return np.inf
    if curva_nome not in CURVAS_DEFS: return np.inf
    k, c, alpha = CURVAS_DEFS[curva_nome]
    val = TMS_TD * ((k) / (M ** alpha - 1.0) + c)
    return max(0.0, val)


# ----------------------------------------------------------------------
# 4. FUNÇÕES DE PLOTAGEM
# ----------------------------------------------------------------------

def gerar_painel_2x2(lista_arquivos_grupo, titulo_grupo, var_name, tipo_grafico, label_y, pasta_saida):
    fig, axs = plt.subplots(2, 2, figsize=(cm_to_inch(20), cm_to_inch(16)))
    axs = axs.flatten()

    # Busca o nome amigável para o título (ex: I_T2F_raw -> Antes do Trafo...)
    nome_amigavel_var = MAPA_LEGENDAS.get(var_name, var_name)

    for i, caminho_str in enumerate(lista_arquivos_grupo):
        ax = axs[i]
        caminho = Path(caminho_str)
        nome_display = formatar_nome_caso(caminho.stem)

        try:
            with h5py.File(caminho, "r") as f:
                t = np.array(f["t"]).flatten()
                dados = padronizar_dados(f, var_name)

                if dados is None or np.max(np.abs(dados)) <= 1e-9:
                    ax.text(0.5, 0.5, "Sem Dados", ha='center', va='center')
                    ax.set_title(nome_display, fontsize=10)
                    continue

                if tipo_grafico == 'RMS':
                    imax = get_imax_envelope_vetor(t, dados)
                    pico = np.max(imax)
                    ax.plot(t, imax, color='black', linewidth=1.5)
                    ax.axvline(x=TEMPO_FALTA, color='red', linestyle='--', linewidth=1.0, alpha=0.8)
                    ax.set_title(f"{nome_display}\nMáx: {pico:.1f}", fontsize=10)
                    ax.set_xlabel("Tempo (s)", fontsize=9)
                    ax.set_ylabel(f"{label_y} (RMS Max)", fontsize=9)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set_ylim(bottom=0)

                elif tipo_grafico == '3FASES':
                    rms_a = calcular_rms_movel(dados[:, 0], t)
                    rms_b = calcular_rms_movel(dados[:, 1], t)
                    rms_c = calcular_rms_movel(dados[:, 2], t)
                    ax.plot(t, rms_a, color='red', linewidth=1.2, alpha=0.9)
                    ax.plot(t, rms_b, color='blue', linewidth=1.2, alpha=0.9)
                    ax.plot(t, rms_c, color='green', linewidth=1.2, alpha=0.9)
                    ax.axvline(x=TEMPO_FALTA, color='black', linestyle='--', linewidth=1.0, alpha=0.6)
                    ax.set_title(f"{nome_display}", fontsize=10)
                    ax.set_xlabel("Tempo (s)", fontsize=9)
                    ax.set_ylabel(f"{label_y}", fontsize=9)
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.set_ylim(bottom=0)

                elif tipo_grafico == 'CLARKE':
                    ia, ib, ic = dados[:, 0], dados[:, 1], dados[:, 2]
                    alpha, beta = clarke_power_invariant(ia, ib, ic)
                    ax.plot(alpha, beta, color='black', linewidth=1.2)
                    ax.set_title(f"{nome_display}", fontsize=10)
                    ax.set_xlabel(r"$i_\alpha$", fontsize=9)
                    ax.set_ylabel(r"$i_\beta$", fontsize=9)
                    ax.axis('equal')
                    ax.grid(True, linestyle='--', alpha=0.5)

        except Exception as e:
            ax.text(0.5, 0.5, "Erro", ha='center')
            print(f"[ERRO] {caminho.name}: {e}")

    # TÍTULO SUPERIOR COM NOME AMIGÁVEL
    plt.suptitle(f"{titulo_grupo} - {tipo_grafico} - {nome_amigavel_var}", fontsize=12)

    # LEGENDA EXTERNA
    if tipo_grafico == '3FASES':
        legend_lines = [
            Line2D([0], [0], color='red', lw=2, label='Fase A'),
            Line2D([0], [0], color='blue', lw=2, label='Fase B'),
            Line2D([0], [0], color='green', lw=2, label='Fase C'),
            Line2D([0], [0], color='black', linestyle='--', lw=1, label='Início Falta')
        ]
        fig.legend(handles=legend_lines, loc='lower center',
                   bbox_to_anchor=(0.5, 0.92), ncol=4, fontsize=9, frameon=False)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
    else:
        plt.tight_layout()

    # Nome arquivo usando nome amigável
    nome_limpo = nome_amigavel_var.replace(" ", "_").replace("ã", "a").replace("ç", "c").replace("õ", "o")
    nome_arquivo_saida = f"PAINEL_{titulo_grupo.replace(' ', '_')}_{tipo_grafico}_{nome_limpo}.svg"
    plt.savefig(pasta_saida / nome_arquivo_saida, format='svg')
    plt.close()


def analisar_protecao_grupo_todas_curvas(lista_arquivos_grupo, titulo_grupo, pasta_saida):
    dados_grupo = []
    pm = RELIGADORES_PARAMS["R_Montante_T2F"]
    pj = RELIGADORES_PARAMS["R_Jusante_T2F1"]

    for caminho_str in lista_arquivos_grupo:
        caminho = Path(caminho_str)
        nome_display = formatar_nome_caso(caminho.stem)

        nome_curto_grafico = nome_display.replace("Com Regulador e Com Terra", "CR+CT") \
            .replace("Sem Regulador e Sem Terra", "SR+ST") \
            .replace("Sem Regulador", "SR").replace("Sem Terra", "ST")

        icc_m_val, icc_j_val = 0.0, 0.0
        try:
            with h5py.File(caminho, "r") as f:
                t = np.array(f["t"]).flatten()
                dm = padronizar_dados(f, "I_T2F_raw")
                if dm is not None: icc_m_val = np.max(get_imax_envelope_vetor(t, dm))
                dj = padronizar_dados(f, "I_T2F1_raw")
                if dj is not None: icc_j_val = np.max(get_imax_envelope_vetor(t, dj))
        except:
            pass

        dados_grupo.append(
            {'nome_display': nome_display, 'nome_curto': nome_curto_grafico, 'icc_m': icc_m_val, 'icc_j': icc_j_val})

    dados_tabela_csv = []
    for nome_curva in CURVAS_DEFS.keys():
        tempos_montante_plot = []
        tempos_jusante_plot = []
        nomes_plot = []

        for caso in dados_grupo:
            tm_val = calcular_tempo_curva(caso['icc_m'] / pm["Pickup"], pm["TMS_TD"], nome_curva) if caso[
                                                                                                         'icc_m'] > 0 else np.inf
            tj_val = calcular_tempo_curva(caso['icc_j'] / pj["Pickup"], pj["TMS_TD"], nome_curva) if caso[
                                                                                                         'icc_j'] > 0 else np.inf
            status = "Coord OK" if (isfinite(tj_val) and tj_val < tm_val) else "---"
            if isfinite(tm_val) and isfinite(tj_val) and tm_val < tj_val: status = "Falha Coord (M < J)"
            if not isfinite(tm_val) and not isfinite(tj_val): status = "Sem Atuação"

            # Use os nomes amigáveis no CSV
            dados_tabela_csv.append([
                nome_curva, caso['nome_display'],
                f"{caso['icc_m']:.2f}", f"{caso['icc_j']:.2f}",
                f"{tm_val:.4f}" if isfinite(tm_val) else "Inf",
                f"{tj_val:.4f}" if isfinite(tj_val) else "Inf",
                status
            ])
            tempos_montante_plot.append(tm_val if isfinite(tm_val) else 0)
            tempos_jusante_plot.append(tj_val if isfinite(tj_val) else 0)
            nomes_plot.append(caso['nome_curto'])

        if nomes_plot:
            x = np.arange(len(nomes_plot))
            width = 0.35
            fig, ax = plt.subplots(figsize=(cm_to_inch(16), cm_to_inch(10)))

            # LEGENDAS ATUALIZADAS NO GRÁFICO DE BARRAS
            rects1 = ax.bar(x - width / 2, tempos_montante_plot, width, label='Antes do Trafo Isolador',
                            color='darkblue')
            rects2 = ax.bar(x + width / 2, tempos_jusante_plot, width, label='Depois do Trafo Isolador',
                            color='darkred')

            ax.set_ylabel('Tempo de Atuação (s)')
            ax.set_title(f'Proteção: {titulo_grupo} - Curva: {nome_curva}')
            ax.set_xticks(x)
            ax.set_xticklabels(nomes_plot, rotation=0, ha='center', fontsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    texto = f'{height:.3f}' if height > 0 else 'Ñ Op.'
                    ax.annotate(texto, xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8,
                                rotation=90)

            autolabel(rects1)
            autolabel(rects2)

            for i in range(len(nomes_plot)):
                h1, h2 = tempos_montante_plot[i], tempos_jusante_plot[i]
                if h1 > 0 and h2 > 0:
                    ax.annotate(f"Δt={abs(h1 - h2):.3f}s", xy=(x[i], max(h1, h2)), xytext=(0, 35),
                                textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

            ymax = max(max(tempos_montante_plot), max(tempos_jusante_plot)) if (
                        tempos_montante_plot or tempos_jusante_plot) else 0
            if ymax > 0: ax.set_ylim(0, ymax * 1.50)

            plt.tight_layout()
            nome_arquivo_saida = f"GRAFICO_PROTECAO_{titulo_grupo.replace(' ', '_')}_{nome_curva}.svg"
            plt.savefig(pasta_saida / nome_arquivo_saida, format='svg')
            plt.close()

    csv_path = pasta_saida / f"TABELA_PROTECAO_{titulo_grupo.replace(' ', '_')}_TODAS_CURVAS.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf, delimiter=';')
        writer.writerow(
            ["Curva", "Caso", "Icc_Antes_Trafo(A)", "Icc_Depois_Trafo(A)", "Tempo_Antes(s)", "Tempo_Depois(s)",
             "Status"])
        writer.writerows(dados_tabela_csv)


# ----------------------------------------------------------------------
# 5. LOOP PRINCIPAL
# ----------------------------------------------------------------------

def main():
    print(f"Iniciando processamento...")
    for i, grupo in enumerate(lista_arquivos_cap5):
        ref_path = Path(grupo[0])
        nome_grupo = identifying_group_by_name(ref_path.stem)

        print(f"--> Grupo: {nome_grupo}")
        pasta_saida = PASTA_RAIZ_RESULTADOS

        # 1. Painel Clarke (I_T2F)
        gerar_painel_2x2(grupo, nome_grupo, "I_T2F_raw", "CLARKE", "Corrente (A)", pasta_saida)
        # 1. Painel Clarke (I_T2F1)
        gerar_painel_2x2(grupo, nome_grupo, "I_T2F1_raw", "CLARKE", "Corrente (A)", pasta_saida)

        # 2. Painel Corrente RMS (I_T2F) com linha vertical
        gerar_painel_2x2(grupo, nome_grupo, "I_T2F_raw", "3FASES", "Corrente (A)", pasta_saida)
        # 2. Painel Corrente RMS (I_T2F1) com linha vertical
        gerar_painel_2x2(grupo, nome_grupo, "I_T2F1_raw", "3FASES", "Corrente (A)", pasta_saida)

        # 3. Painel Tensão RMS (V_T2F) com linha vertical
        gerar_painel_2x2(grupo, nome_grupo, "V_T2F_raw", "3FASES", "Tensão (V)", pasta_saida)
        # 3. Painel Tensão RMS (V_T2F) com linha vertical
        gerar_painel_2x2(grupo, nome_grupo, "V_T2F1_raw", "3FASES", "Tensão (V)", pasta_saida)

        # 4. Análise de Proteção (TODAS AS CURVAS)
        analisar_protecao_grupo_todas_curvas(grupo, nome_grupo, pasta_saida)

    print(f"\n--- Finalizado! Resultados em: {PASTA_RAIZ_RESULTADOS.absolute()}")

# CORREÇÃO DA INDENTAÇÃO FINAL
if __name__ == "__main__":
    main()