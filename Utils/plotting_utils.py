import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import datetime
import numpy as np
import pandas as pd


def setup_results_dir(trajectory_name, test_type):
    """
    Crea una struttura gerarchica: Grafici / <Traiettoria> / <Tipo_Test> / Run_<Ora>
    """
    timestamp = datetime.datetime.now().strftime("%H%M%S")

    # Percorso: Grafici / Pista_Racing / Analisi_Nominale / Run_123456
    base_path = os.path.join("Grafici", trajectory_name, test_type)
    run_folder = f"Run_{timestamp}"
    run_dir = os.path.join(base_path, run_folder)

    os.makedirs(run_dir, exist_ok=True)
    print(f"\n--> Risultati pronti in: {run_dir}")
    return run_dir


def save_simulation_data(run_dir, history):
    """Salva i dati in CSV per analisi numeriche (Excel/Matlab)"""
    df = pd.DataFrame(history)
    df.to_csv(os.path.join(run_dir, "dati_analitici.csv"), index=False)


def save_metadata(run_dir, params, stats):
    """Salva i parametri di setup e i risultati di performance (RMSE)"""
    with open(os.path.join(run_dir, "info_simulazione.txt"), "w") as f:
        f.write("=== CONFIGURAZIONE SIMULAZIONE ===\n")
        for k, v in params.items(): f.write(f"- {k}: {v}\n")
        f.write("\n=== PERFORMANCE TRACKING ===\n")
        for k, v in stats.items(): f.write(f"- {k}: {v}\n")


def plot_dashboard(run_dir, history, path, title_suffix=""):
    """
    Genera il dashboard a 4 subplot, lo salva e lo MOSTRA a video.
    """
    time_axis = np.linspace(0, 30, len(history['e']))
    mode_map = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}
    mode_nums = [mode_map[m] for m in history['mode']]

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(3, 2)

    # --- 1. MAPPA TRAIETTORIA (Sinistra) ---
    ax_map = fig.add_subplot(gs[:, 0])
    rx, ry = zip(*path)
    ax_map.plot(rx, ry, 'r--', label='Riferimento', alpha=0.5)
    ax_map.plot(history['x'], history['y'], 'b-', label='Veicolo', linewidth=2)
    ax_map.set_title(f"Percorso Veicolo {title_suffix}", fontweight='bold')
    ax_map.set_xlabel("X [m]")
    ax_map.set_ylabel("Y [m]")
    ax_map.axis('equal')
    ax_map.grid(True, alpha=0.3)
    ax_map.legend()

    # --- 2. ERRORI (Destra Alto) ---
    ax_err = fig.add_subplot(gs[0, 1])
    ax_err.plot(time_axis, history['e'], 'b-', label='Errore Laterale $e$ [m]')
    ax_err.plot(time_axis, history['theta_e'], 'm--', label='Errore Theta $\\theta_e$ [rad]', alpha=0.6)
    ax_err.set_title("Stato degli Errori")
    ax_err.grid(True, alpha=0.3)
    ax_err.legend()

    # --- 3. VELOCITÀ LPV (Destra Centro) ---
    ax_vx = fig.add_subplot(gs[1, 1], sharex=ax_err)
    ax_vx.plot(time_axis, history['vx'], 'k-', label='Velocità Reale $v_x$')
    ax_vx.axhline(y=1.0, color='g', linestyle=':', label='Soglia MED')
    ax_vx.axhline(y=2.2, color='r', linestyle=':', label='Soglia HIGH')
    ax_vx.set_ylabel("[m/s]")
    ax_vx.set_title("Parametro LPV di Scheduling")
    ax_vx.grid(True, alpha=0.3)
    ax_vx.legend(loc='lower right')

    # --- 4. STATO SUPERVISORE (Destra Basso) ---
    ax_sw = fig.add_subplot(gs[2, 1], sharex=ax_err)
    ax_sw.step(time_axis, mode_nums, 'g-', where='post', linewidth=2)
    ax_sw.set_yticks([1, 2, 3])
    ax_sw.set_yticklabels(['LOW', 'MED', 'HIGH'])
    ax_sw.set_ylabel("Modo")
    ax_sw.set_xlabel("Tempo [s]")
    ax_sw.set_title("Stato del Supervisore S")
    ax_sw.grid(True, alpha=0.3)

    plt.tight_layout()

    # Salvataggio
    save_path = os.path.join(run_dir, "Dashboard_Completa.png")
    plt.savefig(save_path, dpi=300)

    # Mostra a schermo (Bloccante)
    print(f"--> Visualizzazione Dashboard in corso... (chiudi la finestra per continuare)")
    plt.show()


def plot_comparison_results(run_dir, error_adaptive, error_fixed, track_name):
    """
    Crea un grafico che sovrappone l'errore del sistema LPV e di quello FISSO.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(error_fixed, 'r--', label='Guadagni Fissi (LOW)', alpha=0.7)
    plt.plot(error_adaptive, 'b-', label='Controllo Adattativo LPV', linewidth=2)

    plt.title(f"Confronto Performance su {track_name}: LPV vs Fisso", fontweight='bold')
    plt.xlabel("Step Temporali")
    plt.ylabel("Errore Laterale [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    save_path = os.path.join(run_dir, "02_Grafico_Confronto.png")
    plt.savefig(save_path, dpi=300)
    print(f"--> Grafico di confronto salvato in: {save_path}")
    plt.show()