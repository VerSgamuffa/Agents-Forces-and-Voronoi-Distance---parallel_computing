import pandas as pd
import matplotlib.pyplot as plt

def main():
    # caricamento dei dati dai file CSV
    try:
        df_seq = pd.read_csv('benchmark_sequenziale.csv')
        df_par = pd.read_csv('benchmark_parallelo.csv')
    except FileNotFoundError as e:
        print("\n[ERRORE] Impossibile trovare i file di benchmark.")
        print(f"Dettaglio errore: {e}")
        return

    # Estrazione e isolamento dei tempi (Voronoi + Forze Attrattive)
    t_seq = df_seq['voronoi_ms'].iloc[0] + df_seq['forces_ms'].iloc[0]

    # calcoliamo la somma per ogni configurazione di thread (parallelo)
    df_par['t_par'] = df_par['voronoi_ms'] + df_par['forces_ms']

    # calcolo dello speedup
    df_par['speedup'] = t_seq / df_par['t_par']

    threads = df_par['threads']

    # aspetto grafici
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['axes.edgecolor'] = '#cccccc'
    plt.rcParams['axes.linewidth'] = 0.8


    # grafico confronto tempi di esecuzione
    plt.figure(figsize=(10, 6))

    plt.plot(threads, df_par['t_par'], marker='o', color='#1f77b4', linewidth=2.5,
             markersize=6, label='Tempo Parallelo ($T_p$)')

    plt.axhline(y=t_seq, color='#d62728', linestyle='--', linewidth=2,
                label='Tempo Sequenziale ($T_{seq}$)')

    plt.title('Confronto Tempi di Calcolo',
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Numero di Thread', fontsize=11, labelpad=8)
    plt.ylabel('Tempo di esecuzione (ms)', fontsize=11, labelpad=8)

    plt.xticks(threads, fontsize=9)
    plt.yticks(fontsize=9)

    plt.grid(True, linestyle=':', alpha=0.6, color='#999999')
    plt.legend(fontsize=10, loc='upper right', frameon=True, facecolor='white', edgecolor='#e5e5e5')
    plt.tight_layout()

    # salvataggio del grafico
    plt.savefig('grafico_confronto_tempi.png', dpi=300)
    plt.close()
    print("-> Generato 'grafico_confronto_tempi.png'")


    #=================================================================================================================
    # grafico speedup
    plt.figure(figsize=(10, 6))


    plt.plot(threads, df_par['speedup'], marker='o', color='#2ca02c', linewidth=2.5,
             markersize=6, label='Speedup Reale ($S_p$)')

    plt.title('Confronto Speedup', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Numero di Thread', fontsize=11, labelpad=8)
    plt.ylabel('Speedup ($T_{seq} / T_p$)', fontsize=11, labelpad=8)

    plt.xticks(threads, fontsize=9)
    plt.yticks(fontsize=9)

    plt.grid(True, linestyle=':', alpha=0.6, color='#999999')
    plt.legend(fontsize=10, loc='upper left', frameon=True, facecolor='white', edgecolor='#e5e5e5')
    plt.tight_layout()

    # salvataggio del secondo grafico
    plt.savefig('grafico_speedup.png', dpi=300)
    plt.close()
    print("-> Generato 'grafico_speedup.png'")

    print("\n[SUCCESSO] Entrambi i grafici sono stati salvati correttamente nella cartella di lavoro!")

if __name__ == '__main__':
    main()