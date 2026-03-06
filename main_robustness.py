import numpy as np
import matplotlib.pyplot as plt
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator, Generator_Noise
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata, plot_dashboard
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_robustness_simulation(scenario_key, usa_rk4=True, live_plot=True):
    # Esegue la simulazione di robustezza con rumore e visualizzazione live.

    # --- CONFIGURAZIONE ---
    # Attiviamo il filtro per gestire meglio il rumore sinusoidale sui 4 livelli
    usa_filtro_supervisore = True
    supervisor_status = "Filtrato" if usa_filtro_supervisore else "Standard"
    integrator_type = "RK4" if usa_rk4 else "Eulero"

    dt = 0.001
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)
    path_array = np.array(path_points)

    # Parametri per la velocità adattativa
    v_max = 3.5
    v_min = 1.0
    sensibilita_curvatura = 12.0

    # Nome del test aggiornato per riflettere l'architettura a 4 livelli
    test_type = f"Robustezza_LPV_4Livelli_{supervisor_status}"
    run_dir = setup_results_dir(track_name, test_type, integrator_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    # Il supervisore utilizzerà la logica a 4 stati (LOW, MED, MED_HIGH, HIGH)
    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # Rumore sui sensori (Incrementato a 0.2 come da tua richiesta)
    noise_magnitude = 0.2
    noise_gen = Generator_Noise(disturb_vx=True,
                                disturb_position=True,
                                magnitude=noise_magnitude,
                                disturbance_type='sinusoidal')

    history = {'x': [], 'y': [], 'vx': [], 'e': [], 'theta_e': [], 'mode': [], 'target_v': []}

    # --- SETUP VISUALIZZAZIONE LIVE OTTIMIZZATA ---
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(path_array[:, 0], path_array[:, 1], 'k--', alpha=0.15)
        line_follower, = ax.plot([], [], 'b-', linewidth=1.5, label='Scia (Reale)')
        current_pos, = ax.plot([], [], 'ro', markersize=6, label='Veicolo')

        ax.set_title(f"Robustness LPV 4 Livelli: {track_name}\n({integrator_type} - {supervisor_status})", fontsize=10)
        ax.grid(True, alpha=0.2)

    print(f"AVVIO ROBUSTEZZA (4 LIVELLI): {track_name} | Rumore Sinusoidale: {noise_magnitude}")

    for i in range(steps):
        t = i * dt

        # 1. Stato Percezione (Dati sporchi dal rumore)
        nvx = state.vx + noise_gen.get_disturbance(t, 'vx')
        nx = state.X + noise_gen.get_disturbance(t, 'position')
        ny = state.Y + noise_gen.get_disturbance(t, 'position')
        # In robustezza manteniamo gli angoli puliti per isolare l'effetto del rumore posizionale/velocità
        perceived = VehicleState(X=nx, Y=ny, phi=state.phi, vx=nvx, vy=state.vy, omega=state.omega)

        # 2. Controllo basato su dati rumorosi e Calcolo Curvatura
        e, theta_e, _, idx = estimator.get_errors(perceived)

        kappa = estimator.get_curvature(idx, lookahead=15)
        target_speed = v_max / (1 + sensibilita_curvatura * kappa)
        target_speed = np.clip(target_speed, v_min, v_max)

        # 3. Supervisore LPV (Utilizzo del filtro consigliato per noise 0.2)
        kp, kd, mode = supervisor.update_and_get_gains(perceived.vx, use_filter=usa_filtro_supervisore)

        # 4. Leggi di Controllo
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speed, state.vx)

        # 5. Fisica Reale (Applichiamo i comandi al modello pulito)
        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()

        try:
            if usa_rk4:
                state = integrator.RK4(state, u_in)
            else:
                state = integrator.Eulero(state, u_in)
        except OverflowError:
            print(f"SISTEMA IN CRUSH a t={t:.2f}s per rumore eccessivo o instabilità numerica")
            break

        # Registrazione dati (Salviamo lo stato REALE del veicolo)
        history['x'].append(state.X)
        history['y'].append(state.Y)
        history['vx'].append(state.vx)
        history['e'].append(e)
        history['theta_e'].append(theta_e)
        history['mode'].append(mode)
        history['target_v'].append(target_speed)

        if live_plot and i % 250 == 0:
            line_follower.set_data(history['x'], history['y'])
            current_pos.set_data([state.X], [state.Y])
            ax.set_xlim(state.X - 8, state.X + 8)
            ax.set_ylim(state.Y - 8, state.Y + 8)
            plt.pause(0.001)

    if live_plot:
        plt.ioff()
        plt.close(fig)

    save_simulation_data(run_dir, history)
    rmse = np.sqrt(np.mean(np.array(history['e']) ** 2))
    save_metadata(run_dir,
                  {"Test": "Robustezza_4Livelli", "Noise": noise_magnitude, "Type": "Sinusoidal", "Int": integrator_type},
                  {"RMSE": f"{rmse:.5f}m", "Status": "Success" if not np.isnan(rmse) else "Failed"})

    # La dashboard mostrerà ora i 4 livelli grazie alla modifica in plotting_utils
    plot_dashboard(run_dir, history, path_points, f"{track_name} - Robustezza 4 Livelli")


if __name__ == "__main__":
    circuito_selezionato = 'racing'
    # Per robustezza con rumore 0.2, RK4 è decisamente preferibile ad Eulero
    run_robustness_simulation(circuito_selezionato, usa_rk4=True, live_plot=True)
    print(f"\n✓ Analisi di robustezza (4 Livelli) su {circuito_selezionato} completata.")