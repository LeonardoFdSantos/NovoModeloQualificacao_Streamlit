import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 0. Configuração
pasta_saida = "Resultados_Tese_Cap5_Sintese"
os.makedirs(pasta_saida, exist_ok=True)

# Definição de estilo para publicação
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.autolayout': True
})

# 1. Carregamento dos Dados
# ==============================================================================
# Ajuste o caminho '../Resultados...' se os arquivos estiverem em outra pasta
caminho_dados = "Resultados_Tese_Cap5_Comparativo_Final"

arquivos = {
    "AB": f"{caminho_dados}/TABELA_PROTECAO_Falta_AB_TODAS_CURVAS.csv",
    "BC": f"{caminho_dados}/TABELA_PROTECAO_Falta_BC_TODAS_CURVAS.csv",
    "CA": f"{caminho_dados}/TABELA_PROTECAO_Falta_AC_TODAS_CURVAS.csv",
    "ABC": f"{caminho_dados}/TABELA_PROTECAO_Falta_ABC_TODAS_CURVAS.csv"
}

dfs = []
for tipo_falta, caminho in arquivos.items():
    try:
        # Tenta carregar. Se falhar, avisa.
        df = pd.read_csv(caminho, sep=';')
        df['Tipo_Falta'] = tipo_falta
        dfs.append(df)
    except FileNotFoundError:
        print(f"Aviso: Arquivo {caminho} não encontrado. Verifique se a pasta anterior foi gerada.")

if not dfs:
    raise ValueError("Nenhum dado encontrado. Rode o script de geração de tabelas primeiro.")

df_geral = pd.concat(dfs, ignore_index=True)

# Tratamento Numérico (Converter 'Inf' para NaN para cálculos de média)
cols_tempo = ['Tempo_Antes(s)', 'Tempo_Depois(s)']
for col in cols_tempo:
    df_geral[col] = pd.to_numeric(df_geral[col], errors='coerce').replace([np.inf, -np.inf], np.nan)

# Cálculo do Delta T para cada linha
df_geral['Delta_T'] = df_geral['Tempo_Antes(s)'] - df_geral['Tempo_Depois(s)']

# 2. Tabela Resumo 1: Tempos Médios de Atuação (Rapidez)
# ==============================================================================
# Agrupa por Curva e Tipo de Falta -> Tira a média dos casos (CR+CT, ST, etc.)
tabela_tempos = df_geral.pivot_table(
    index='Curva',
    columns='Tipo_Falta',
    values='Tempo_Depois(s)',
    aggfunc='mean'
)

# Calcula média global (todas as faltas)
tabela_tempos['t_medio_global'] = tabela_tempos.mean(axis=1)
# Ordena da mais rápida para a mais lenta
tabela_tempos = tabela_tempos.sort_values('t_medio_global')

# Salva CSV
caminho_tabela_tempos = f"{pasta_saida}/Tabela_Sintese_Tempos_Atuacao.csv"
tabela_tempos.to_csv(caminho_tabela_tempos, float_format='%.4f')

# 3. Tabela Resumo 2: Margens Médias de Coordenação (Seletividade)
# ==============================================================================
tabela_margens = df_geral.pivot_table(
    index='Curva',
    columns='Tipo_Falta',
    values='Delta_T',
    aggfunc='mean'
)

tabela_margens['dt_medio_global'] = tabela_margens.mean(axis=1)
# Reordena para ficar igual à tabela de tempos
tabela_margens = tabela_margens.reindex(tabela_tempos.index)

# Salva CSV
caminho_tabela_margens = f"{pasta_saida}/Tabela_Sintese_Margens_Coordenacao.csv"
tabela_margens.to_csv(caminho_tabela_margens, float_format='%.4f')

# 4. Gráfico Sintético Final (Trade-off Rapidez x Seletividade)
# ==============================================================================
fig, ax1 = plt.subplots(figsize=(12, 7))

curvas = tabela_tempos.index
t_medios = tabela_tempos['t_medio_global']
dt_medios = tabela_margens['dt_medio_global']

# Série 1: Tempo Médio (Barras Azuis)
bars = ax1.bar(curvas, t_medios, color='steelblue', edgecolor='black', alpha=0.7,
               label='Tempo Médio de Atuação ($t_{op}$)')
ax1.set_xlabel('Curva TCC', fontsize=12)
ax1.set_ylabel('Tempo Médio de Atuação (s)', color='steelblue', fontsize=12)
ax1.tick_params(axis='y', labelcolor='steelblue')
ax1.set_ylim(0, max(t_medios) * 1.3)  # Margem superior para legenda

# Série 2: Margem de Coordenação (Linha Vermelha com Eixo Secundário)
ax2 = ax1.twinx()
ax2.plot(curvas, dt_medios, color='darkred', marker='D', linewidth=2, markersize=8, label='Margem Média ($\Delta t$)')
ax2.set_ylabel('Margem de Coordenação Média (s)', color='darkred', fontsize=12)
ax2.tick_params(axis='y', labelcolor='darkred')

# Linhas de referência
ax2.axhline(0, color='gray', linestyle='-', linewidth=1)
ax2.axhline(0.2, color='green', linestyle='--', linewidth=1.5, label='Meta Mínima (0.2s)')

# Legenda Unificada
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc='upper left', frameon=True)

plt.title('Síntese Final: Rapidez vs. Seletividade das Curvas TCC')
plt.savefig(f"{pasta_saida}/Grafico_Sintese_TradeOff.svg")
plt.close()

# 5. Critério de Escolha e Relatório Automático
# ==============================================================================
# Verifica falhas: conta quantos cenários individuais têm Delta T < 0
falhas_por_curva = df_geral[df_geral['Delta_T'] < 0].groupby('Curva').size()
total_cenarios = len(df_geral) / len(tabela_tempos)  # Cenários por curva
pct_falhas = (falhas_por_curva / total_cenarios * 100).fillna(0)

print("\n=======================================================")
print(" RELATÓRIO DE SÍNTESE E ESCOLHA DA MELHOR CURVA")
print("=======================================================")

print("\n1. Tabela de Tempos Médios (s):")
print(tabela_tempos['t_medio_global'].to_string())

print("\n2. Tabela de Margens Médias (s):")
print(tabela_margens['dt_medio_global'].to_string())

# Lógica de Decisão
print("\n3. Veredito:")
curvas_sem_falha = pct_falhas[pct_falhas == 0].index.tolist()

if curvas_sem_falha:
    print(f"-> Curvas que garantem seletividade em 100% dos casos: {curvas_sem_falha}")
    # Entre as que passam, pega a mais rápida
    melhor_curva = tabela_tempos.loc[curvas_sem_falha, 't_medio_global'].idxmin()
    tempo_melhor = tabela_tempos.loc[melhor_curva, 't_medio_global']
    print(f"-> MELHOR ESCOLHA: {melhor_curva} (Tempo médio: {tempo_melhor:.3f}s)")
else:
    print("-> Nenhuma curva garantiu seletividade em 100% dos casos simulados.")
    print("-> Percentual de cenários com falha por curva:")
    print(pct_falhas.sort_values().to_string())

    # Alternativa: Curva com melhor margem média (mesmo que negativa)
    melhor_margem_nome = tabela_margens['dt_medio_global'].idxmax()
    melhor_margem_val = tabela_margens.loc[melhor_margem_nome, 'dt_medio_global']

    print(f"\n-> Recomendação (Maior Robustez): {melhor_margem_nome}")
    print(f"   Apresentou a melhor margem média ({melhor_margem_val:.3f}s), indicando ser a opção")
    print(f"   que mais se aproxima da coordenação ideal, apesar das falhas.")

print(f"\nArquivos gerados na pasta '{pasta_saida}'")