import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import datetime
import numpy as np


def setup_results_dir(phase_folder, trajectory_name, param_string):

    timestamp = datetime.datetime.now().strftime("%H%M%S")
    base_path = os.path.join("Risultati_Tesi", phase_folder)
    folder_name = f"{trajectory_name}_{param_string}_{timestamp}"
    run_dir = os.path.join(base_path, folder_name)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_metadata(run_dir, params_dict, stats_dict):

    with open(os.path.join(run_dir, "info_simulazione.txt"), "w") as f:
        f.write("PARAMETRI CONFIGURATI:\n")
        for k, v in params_dict.items(): f.write(f"- {k}: {v}\n")
        f.write("\nRISULTATI OTTENUTI:\n")
        for k, v in stats_dict.items(): f.write(f"- {k}: {v}\n")


def plot_sim_results(run_dir, time, data_dict, ref_path, traj_x, traj_y):
    # Grafico XY
    plt.figure(figsize=(10, 8))
    rx, ry = zip(*ref_path)
    plt.plot(rx, ry, 'r--', label='Riferimento')
    plt.plot(traj_x, traj_y, 'b-', label='Seguita')
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(run_dir, "01_Mappa_Traiettoria.png"), dpi=300)
    plt.close()

    # Grafici stati
    num = len(data_dict)
    fig, axes = plt.subplots(num, 1, figsize=(12, 3 * num), sharex=True)
    if num == 1: axes = [axes]
    for i, (label, info) in enumerate(data_dict.items()):
        axes[i].plot(time, info['val'], label='Reale')
        if 'ref' in info:
            # Se il riferimento è una lista o un array (come omega_des)
            if isinstance(info['ref'], (list, np.ndarray)):
                axes[i].plot(time, info['ref'], 'r--', alpha=0.8, label='Target')
            # Se il riferimento è un numero singolo (come vx_des)
            else:
                axes[i].axhline(y=info['ref'], color='r', linestyle='--', label='Target')
        axes[i].set_ylabel(label)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "02_Grafici_Stati.png"), dpi=300)
    plt.close()


def genera_video_animazione(ref_path, traj_x, traj_y, dt, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    rx, ry = zip(*ref_path)
    ax.plot(rx, ry, 'r--', label='Riferimento')
    line, = ax.plot([], [], 'b-', label='Veicolo')
    point, = ax.plot([], [], 'go')
    ax.set_xlim(min(rx) - 1, max(rx) + 1)
    ax.set_ylim(min(ry) - 1, max(ry) + 1)
    ax.set_aspect('equal')

    def update(f):
        line.set_data(traj_x[:f], traj_y[:f])
        point.set_data([traj_x[f]], [traj_y[f]])
        return line, point

    ani = FuncAnimation(fig, update, frames=range(0, len(traj_x), 20), interval=dt * 20 * 1000, blit=True)
    plt.show()


def plot_disturbance_analysis(run_dir, time_history, vx_history, omega_history,
                              vx_disturb_history, omega_disturb_history,
                              phase_name, vx_controlled=False, omega_controlled=False):

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Disturbo su vx
    axes[0].plot(time_history, vx_disturb_history, 'red', linewidth=1.5, alpha=0.7)
    axes[0].set_ylabel('Disturbo vx [m/s]', fontsize=11)
    axes[0].set_title('Disturbo applicato su velocità longitudinale', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Subplot 2: vx con disturbo
    control_status = "CONTROLLATA" if vx_controlled else "NON controllata"
    axes[1].plot(time_history, vx_history, 'b-', linewidth=2, label=f'vx (con disturbo)')
    axes[1].axhline(y=np.mean(vx_history), color='gray', linestyle=':',
                    linewidth=1.5, label=f'Media: {np.mean(vx_history):.2f} m/s')
    axes[1].set_ylabel('vx [m/s]', fontsize=11)
    axes[1].set_title(f'Velocità longitudinale - {phase_name} ({control_status})',
                      fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Subplot 3: Disturbo su omega
    axes[2].plot(time_history, omega_disturb_history, 'red', linewidth=1.5, alpha=0.7)
    axes[2].set_ylabel('Disturbo ω [rad/s]', fontsize=11)
    axes[2].set_title('Disturbo applicato su velocità angolare', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Subplot 4: omega con disturbo
    control_status_omega = "CONTROLLATA" if omega_controlled else "NON controllata"
    axes[3].plot(time_history, omega_history, 'purple', linewidth=2, label='ω (con disturbo)')
    axes[3].set_ylabel('ω [rad/s]', fontsize=11)
    axes[3].set_xlabel('Tempo [s]', fontsize=12)
    axes[3].set_title(f'Velocità angolare - {phase_name} ({control_status_omega})',
                      fontsize=12, fontweight='bold')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "03_Analisi_Disturbi.png"), dpi=300)
    plt.close()
