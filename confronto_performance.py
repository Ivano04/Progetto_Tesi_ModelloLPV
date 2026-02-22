import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator, Generator_Noise
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_simulation(track_type, integrator_type="RK4", with_noise=False):

    #Esegue una simulazione completa con i parametri specificati e restituisce l'array degli errori.

    dt = 0.001
    total_time = 25.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(track_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

    # Inizializzazione stato
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    supervisor = SupervisorS()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # Configurazione Rumore (Magnitudo 0.07 come nei test di robustezza)
    noise_gen = Generator_Noise(disturb_vx=True, disturb_position=True, magnitude=0.07) if with_noise else None

    target_speeds = np.linspace(0.5, 3.5, steps)
    history_e = []

    for i in range(steps):
        t = i * dt

        # 1. Percezione (Nominale o con Rumore)
        if with_noise:
            nvx = state.vx + noise_gen.get_disturbance(t, 'vx')
            nx = state.X + noise_gen.get_disturbance(t, 'position')
            ny = state.Y + noise_gen.get_disturbance(t, 'position')
            perceived = VehicleState(X=nx, Y=ny, phi=state.phi, vx=nvx, vy=state.vy, omega=state.omega)
        else:
            perceived = state

        # 2. Controllo LPV (Attiviamo il filtro se c'è rumore per un confronto equo)
        e, theta_e, _ = estimator.get_errors(perceived)
        kp, kd, _ = supervisor.update_and_get_gains(perceived.vx, use_filter=with_noise)

        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speeds[i], state.vx)

        # 3. Applicazione Input e Integrazione
        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()

        if integrator_type == "RK4":
            state = integrator.RK4(state, u_in)
        else:
            state = integrator.Eulero(state, u_in)

        history_e.append(e)

        # Check divergenza (se il modello esplode, interrompiamo per evitare attese inutili)
        if np.isnan(state.vx) or abs(state.vx) > 50:
            # Riempiamo il resto con NaN per segnalare il fallimento nel grafico
            history_e.extend([np.nan] * (steps - len(history_e)))
            break

    return np.array(history_e), track_name


def plot_integrator_comparison(run_dir, err_rk4, err_euler, track_name, noise_status):
    #Genera un grafico specifico per il confronto tra i due metodi di integrazione.
    plt.figure(figsize=(12, 6))

    # Plot Eulero (verifichiamo se è esploso)
    if np.any(np.isnan(err_euler)):
        plt.text(0.5, 0.5, 'DIVERGENZA NUMERICA EULERO',
                 horizontalalignment='center', verticalalignment='center',
                 transform=plt.gca().transAxes, fontsize=14, color='red', fontweight='bold')
    else:
        plt.plot(err_euler, 'r--', label='Eulero (1° Ordine)', alpha=0.7)

    # Plot RK4
    plt.plot(err_rk4, 'b-', label='RK4 (4° Ordine)', linewidth=2)

    plt.title(f"Confronto Integratori: {track_name} ({noise_status})", fontweight='bold')
    plt.xlabel("Step Temporali")
    plt.ylabel("Errore Laterale [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(os.path.join(run_dir, "Confronto_Eulero_vs_RK4.png"), dpi=300)
    plt.close()


def main_confronto_integratori():
    circuits = ['racing', 'circular', 'eight']
    noise_scenarios = [False, True]

    print("=== AVVIO ANALISI COMPARATIVA: EULERO VS RK4 ===")

    for scenario in circuits:
        for with_noise in noise_scenarios:
            noise_label = "Con_Rumore" if with_noise else "Senza_Rumore"
            print(f"\n> Analisi su: {scenario} | {noise_label}")

            # Esecuzione RK4
            err_rk4, track_name = run_simulation(scenario, "RK4", with_noise)
            # Esecuzione Eulero
            err_euler, _ = run_simulation(scenario, "Eulero", with_noise)

            # Setup cartella: Grafici_Confronto_Integratori / <Pista> / <Noise> / Run_...
            # Passiamo "Confronto_Integratori" come tipo per creare la root corretta
            run_dir = setup_results_dir(track_name, noise_label, "Confronto_Integratori")

            # Calcolo RMSE (solo se non ci sono NaN)
            rmse_rk4 = np.sqrt(np.mean(err_rk4 ** 2))
            is_euler_stable = not np.any(np.isnan(err_euler))
            rmse_euler = np.sqrt(np.mean(err_euler[~np.isnan(err_euler)] ** 2)) if is_euler_stable else float('inf')

            # Salvataggio Metadati
            stats = {
                "RMSE_RK4": f"{rmse_rk4:.6f} m",
                "RMSE_Eulero": f"{rmse_euler:.6f} m" if is_euler_stable else "DIVERGENZA",
                "Stato_Numerico_Eulero": "STABILE" if is_euler_stable else "FALLITO (Overflow)",
                "Presenza_Rumore": "Sì (0.07)" if with_noise else "No"
            }
            save_metadata(run_dir, {"Pista": track_name, "Condizione": noise_label}, stats)

            # Salvataggio CSV e Plot
            save_simulation_data(run_dir, {"err_rk4": err_rk4, "err_euler": err_euler})
            plot_integrator_comparison(run_dir, err_rk4, err_euler, track_name, noise_label)

    print("\n✓ CONFRONTO COMPLETATO. Tutti i dati sono in 'Grafici_Confronto_Integratori/'.")


if __name__ == "__main__":
    main_confronto_integratori()