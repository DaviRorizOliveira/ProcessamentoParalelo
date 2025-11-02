import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def analiseDados():
    """
    Função principal que orquestra a análise.
    """
    # A função Tabelas agora processa os dados e retorna um
    # DataFrame único e limpo, pronto para plotar.
    df_analysis = Tabelas()
    
    # Se a leitura e processamento dos dados for bem-sucedida, gera os gráficos
    if df_analysis is not None:
        Graficos(df_analysis)

def escreveTabelas(df):
    """
    Recebe o DataFrame de análise final e o salva em um arquivo de texto.
    """
    # Garante que o diretório 'tabelas' exista (adicionado para robustez)
    os.makedirs("tabelas", exist_ok=True)
    
    with open("tabelas/tabelas.txt", 'w') as arq :
        arq.write("Analise Comparativa (Media OMP/Seq vs MPI 2, 4, 8)\n\n")
        # .to_string() garante que o DataFrame completo seja salvo
        # Define 'tamMatriz' como índice para uma impressão mais limpa
        arq.write(df.set_index('tamMatriz').to_string())

def Tabelas():
    """
    Lê o CSV bruto, calcula as médias de OMP/Seq e separa
    os 3 resultados de MPI com base na ordem de execução.
    """
    try:
        df_raw = pd.read_csv("resultados.csv")
    except FileNotFoundError:
        print("ERRO: Arquivo 'resultados.csv' nao encontrado.")
        print("Por favor, execute o ./main primeiro.")
        return None
    except pd.errors.EmptyDataError:
        print("ERRO: O arquivo 'resultados.csv' esta vazio.")
        return None

    # Agrupa os dados por tamanho da matriz
    grouped = df_raw.groupby('tamMatriz')
    
    # CALCULA A MÉDIA dos dados Sequencial e OpenMP
    # Lista de colunas para calcular a média
    omp_seq_cols = [
        'tempoSequencial', 
        'tempo2Thread', 'speedup2Thread', 'eficiencia2Thread',
        'tempo4Thread', 'speedup4Thread', 'eficiencia4Thread',
        'tempo8Thread', 'speedup8Thread', 'eficiencia8Thread'
    ]
    
    # .mean() calcula a média para cada 'tamMatriz'
    # reset_index() transforma o 'tamMatriz' de volta em uma coluna
    df_analysis = grouped[omp_seq_cols].mean().reset_index()
    
    # ---
    # 2. EXTRAI OS DADOS MPI (baseado na ORDEM DE EXECUÇÃO)
    #    Conforme a sua garantia:
    #    - nth(0) = primeira execução = np 2
    #    - nth(1) = segunda execução  = np 4
    #    - nth(2) = terceira execução = np 8
    
    mpi_cols_with_key = ['tamMatriz', 'tempoMPI', 'speedupMPI', 'eficienciaMPI']
    
    try:
        # Pega a 1ª linha de cada grupo e renomeia as colunas
        df_mpi_np2 = grouped.nth(0)[mpi_cols_with_key].rename(columns={
            'tempoMPI': 'tempoMPI_2',
            'speedupMPI': 'speedupMPI_2',
            'eficienciaMPI': 'eficienciaMPI_2'
        })
        
        # Pega a 2ª linha de cada grupo
        df_mpi_np4 = grouped.nth(1)[mpi_cols_with_key].rename(columns={
            'tempoMPI': 'tempoMPI_4',
            'speedupMPI': 'speedupMPI_4',
            'eficienciaMPI': 'eficienciaMPI_4'
        })
        
        # Pega a 3ª linha de cada grupo
        df_mpi_np8 = grouped.nth(2)[mpi_cols_with_key].rename(columns={
            'tempoMPI': 'tempoMPI_8',
            'speedupMPI': 'speedupMPI_8',
            'eficienciaMPI': 'eficienciaMPI_8'
        }) # Remove .reset_index()
        
    except IndexError:
        print(f"Verifique 'resultados.csv'. Esperado 3 linhas por 'tamMatriz', mas nao foram encontradas.")
        print("Certifique-se de que executou ./main com -np 2, -np 4 e -np 8.")
        return None

    # 3. JUNTA TUDO: Usa pd.merge (mais robusto) para juntar
    #    os DataFrames usando a coluna 'tamMatriz'
    df_final = pd.merge(df_analysis, df_mpi_np2, on='tamMatriz')
    df_final = pd.merge(df_final, df_mpi_np4, on='tamMatriz')
    df_final = pd.merge(df_final, df_mpi_np8, on='tamMatriz')
    
    # Salva o novo DataFrame consolidado nas tabelas
    escreveTabelas(df_final)

    return df_final

def Graficos(df):
    """
    Recebe o DataFrame de análise (com médias OMP e dados MPI separados)
    e plota os 3 gráficos.
    """
    
    # 'tamMatriz' agora é uma coluna, não o índice
    tamMatrizes = df['tamMatriz']

    # Garante que o diretório 'graficos' exista (adicionado para robustez)
    os.makedirs("graficos", exist_ok=True)
    plt.style.use('seaborn-v0_8')

# ============ GRÁFICO 1: Tempos de Execução ============
    plt.figure(figsize = (12, 8)) 
    
    # Dados Sequencial e OpenMP (MÉDIAS)
    plt.plot(tamMatrizes, df['tempoSequencial'], 'o-', linewidth=2, markersize=8, label='Sequencial (Média)', color='#1f77b4')
    plt.plot(tamMatrizes, df['tempo2Thread'], 's-', markersize=6, label='OMP 2 Threads (Média)', color='#ff7f0e')
    plt.plot(tamMatrizes, df['tempo4Thread'], '^-', markersize=6, label='OMP 4 Threads (Média)', color='#2ca02c')
    plt.plot(tamMatrizes, df['tempo8Thread'], 'd-', markersize=6, label='OMP 8 Threads (Média)', color='#d62728')
    
    # Dados MPI (Individuais, baseados na ordem de execução)
    plt.plot(tamMatrizes, df['tempoMPI_2'], 'x--', markersize=6, label='MPI (2 Procs)', color='#9467bd')
    plt.plot(tamMatrizes, df['tempoMPI_4'], 'P--', markersize=6, label='MPI (4 Procs)', color='#8c564b')
    plt.plot(tamMatrizes, df['tempoMPI_8'], '*--', markersize=6, label='MPI (8 Procs)', color='#e377c2')

    plt.xticks(tamMatrizes, [f"{int(x)}" for x in tamMatrizes])
    plt.xlabel('Tamanho da Matriz', fontsize=12, fontweight='bold')
    plt.ylabel('Tempo (segundos)', fontsize=12, fontweight='bold')
    plt.title('Tempo de Execução (Média OMP/Seq vs MPI)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graficos/graficoExecucao_dgemm.png', dpi=300, bbox_inches='tight')
    plt.close()

 # ========== GRÁFICO 2: SPEEDUP ==========
    plt.figure(figsize = (12, 8))
    
    # Dados OpenMP (MÉDIAS)
    plt.plot(tamMatrizes, df['speedup2Thread'], 's-', markersize=6, label='OMP 2 Threads (Média)', color='#ff7f0e')
    plt.plot(tamMatrizes, df['speedup4Thread'], '^-', markersize=6, label='OMP 4 Threads (Média)', color='#2ca02c')
    plt.plot(tamMatrizes, df['speedup8Thread'], 'd-', markersize=6, label='OMP 8 Threads (Média)', color='#d62728')
    
    # Dados MPI (Individuais)
    plt.plot(tamMatrizes, df['speedupMPI_2'], 'x--', markersize=6, label='MPI (2 Procs)', color='#9467bd')
    plt.plot(tamMatrizes, df['speedupMPI_4'], 'P--', markersize=6, label='MPI (4 Procs)', color='#8c564b')
    plt.plot(tamMatrizes, df['speedupMPI_8'], '*--', markersize=6, label='MPI (8 Procs)', color='#e377c2')

    plt.xticks(tamMatrizes, [f"{int(x)}" for x in tamMatrizes])
    
    # Linhas de referência (speedup ideal)
    plt.axhline(y=2, color='#ff7f0e', linestyle='--', alpha=0.3)
    plt.axhline(y=4, color='#2ca02c', linestyle='--', alpha=0.3)
    plt.axhline(y=8, color='#d62728', linestyle='--', alpha=0.3)
    
    plt.xlabel('Tamanho da Matriz', fontsize=12, fontweight='bold')
    plt.ylabel('Speedup', fontsize=12, fontweight='bold')
    plt.title('Speedup (Média OMP/Seq vs MPI)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graficos/graficoSpeedup_dgemm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========== GRÁFICO 3: EFICIÊNCIA ==========
    plt.figure(figsize=(12, 8))
    
    # Dados OpenMP (MÉDIAS)
    plt.plot(tamMatrizes, df['eficiencia2Thread'], 's-', markersize=6, label='OMP 2 Threads (Média)', color='#ff7f0e')
    plt.plot(tamMatrizes, df['eficiencia4Thread'], '^-', markersize=6, label='OMP 4 Threads (Média)', color='#2ca02c')
    plt.plot(tamMatrizes, df['eficiencia8Thread'], 'd-', markersize=6, label='OMP 8 Threads (Média)', color='#d62728')

    # Dados MPI (Individuais)
    plt.plot(tamMatrizes, df['eficienciaMPI_2'], 'x--', markersize=6, label='MPI (2 Procs)', color='#9467bd')
    plt.plot(tamMatrizes, df['eficienciaMPI_4'], 'P--', markersize=6, label='MPI (4 Procs)', color='#8c564b')
    plt.plot(tamMatrizes, df['eficienciaMPI_8'], '*--', markersize=6, label='MPI (8 Procs)', color='#e377c2')

    plt.xticks(tamMatrizes, [f"{int(x)}" for x in tamMatrizes])
    
    # Linha de referência (eficiência ideal)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Eficiência Ideal (100%)')
    
    plt.xlabel('Tamanho da Matriz', fontsize=12, fontweight='bold')
    plt.ylabel('Eficiência', fontsize=12, fontweight='bold')
    plt.title('Eficiência (Média OMP/Seq vs MPI)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('graficos/graficoEficiencia_dgemm.png', dpi=300, bbox_inches='tight')
    plt.close()

def main ():
    analiseDados()

if __name__ == "__main__":
    main()

