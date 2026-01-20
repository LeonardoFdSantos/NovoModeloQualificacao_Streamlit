"""
================================================================================
ANÁLISE CAPÍTULO 4 - TESTES DE CURTO-CIRCUITO
Sistema IEEE 34 Barras - MRT (SWER) vs T2F
Versão Profissional para Tese de Doutorado
================================================================================
Autor: Leonardo Santos
Data: Janeiro 2026
================================================================================
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- CONFIGURAÇÕES ---
# 1. Qual arquivo você vai ler?
caminho_arquivo = Path('C:/Users/leosa/OneDrive/Coisas_Leonardo/gits/CurtosT2F/T2F_MATLAB/NovoArtigoPowerDelivery34bus/NovoModeloQualificacao/Teste_Novo_Sem_Terra_14/Processados_HDF5/MRT__A_818_1_-_Falta_A_py.mat')

# Pasta onde tudo será salvo (Raiz)
pasta_raiz = Path('Resultados_Tese')

# --- 2.FUNÇÃO RMS(Otimizada) - --
def calcular_rms_movel(sinal, tempo, freq_rede=60):
    dt = tempo[1] - tempo[0]
    fs = 1 / dt
    janela = int(fs / freq_rede)

    # Cálculo rápido via convolução
    sinal_quadrado = sinal ** 2
    janela_media = np.ones(janela) / janela
    # mode='same' mantém o tamanho do vetor igual ao do tempo
    return np.sqrt(np.convolve(sinal_quadrado, janela_media, mode='same'))


# --- 3. LÓGICA DE PASTAS E PLOTAGEM ---
try:
    # Passo A: Criar a pasta com o NOME DO ARQUIVO
    nome_da_pasta = caminho_arquivo.stem  # Pega "seu_arquivo_v73" sem o .mat
    pasta_final = pasta_raiz / nome_da_pasta

    # Cria a pasta (se já existir, não dá erro)
    pasta_final.mkdir(parents=True, exist_ok=True)
    print(f"--> Criando pasta e salvando em: {pasta_final}")

    # Passo B: Ler e Calcular
    with h5py.File(caminho_arquivo, 'r') as f:
        t = np.array(f['t']).flatten()
        # Transposta (.T) porque o h5py lê invertido
        dados = np.array(f['V_800_raw']).T
        fase_a = dados[:, 0]
        fase_b = dados[:, 1]
        fase_c = dados[:, 2]

        print("Calculando RMS...")
        rms = calcular_rms_movel(fase_a, t)

        # Passo C: Plotar
        # Configuração ABNT (Times New Roman, 16cm)
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times New Roman"],
            "font.size": 12, "figure.figsize": (16 / 2.54, 10 / 2.54)
        })

        plt.figure()
        plt.plot(t, rms, color='red', label='RMS Fase A')
        plt.xlabel('Tempo (s)')
        plt.ylabel('Tensão (V)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()

        # Passo D: SALVAR DENTRO DA PASTA ESPECÍFICA
        nome_imagem = pasta_final / 'tensao_rms.svg'
        plt.savefig(nome_imagem, format='svg')

        print(f"Sucesso! Gráfico salvo em: {nome_imagem}")
        plt.close()  # Fecha a figura para liberar memória

except Exception as e:
    print(f"Erro: {e}")