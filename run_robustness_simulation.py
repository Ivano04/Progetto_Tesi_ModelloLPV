import numpy as np
import matplotlib.pyplot as plt
import os
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator, Generator_Noise
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata, plot_dashboard
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_robustness_simulation(scenario_key, usa_rk4=True, live_plot=True):
    # --- CONFIGURAZIONE ---
    usa_filtro_supervisore = True  # Meglio attivarlo se il rumore è alto (0.97)
    integrator_type = "RK4" if usa_rk4 else "Eulero"

    dt = 0.004  # Portato a 0.01: RK4 è stabile e la simulazione è 10x più veloce
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)
    path_array = np.array(path_points)

    # Parametri Velocità Adattativa
    v_max = 3.5
    v_min = 1.0
    sensibilita_curvatura = 12.0

    run_dir = setup_results_dir(track_name, f"Test_Sbandamento_Adattativo", integrator_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # --- RUMORE ---
    noise_magnitude = 0.07
    noise_gen = Generator_Noise(
        disturb_vx=True,
        disturb_position=True,
        disturb_heading=True,
        magnitude=noise_magnitude,
        magnitude_position=noise_magnitude,
        magnitude_heading=noise_magnitude * 0.97,
        disturbance_type="noise"
    )

    history = {'x': [], 'y': [], 'vx': [], 'e': [], 'theta_e': [], 'mode': [], 'target_v': []}
    last_noise = {'x': 0, 'y': 0, 'phi': 0, 'vx': 0}

    if live_plot:
        plt.ion()  # Modalità interattiva ON
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(path_array[:, 0], path_array[:, 1], 'k--', alpha=0.15)
        line_follower, = ax.plot([], [], 'b-', linewidth=1.5, label='Traiettoria Reale')
        current_pos, = ax.plot([], [], 'ro', markersize=6)
        sensor_pos, = ax.plot([], [], 'gx', markersize=4, alpha=0.3, label='Percezione Rumorosa')
        ax.legend()

    print(f"AVVIO TEST: {track_name} | Rumore: {noise_magnitude} | Integratore: {integrator_type}")

    for i in range(steps):
        t = i * dt

        if i % 10 == 0:
            last_noise['vx'] = noise_gen.get_disturbance(t, 'vx')
            last_noise['x'] = noise_gen.get_disturbance(t, 'position')
            last_noise['y'] = noise_gen.get_disturbance(t, 'position')
            last_noise['phi'] = noise_gen.get_disturbance(t, 'heading')

        nvx = state.vx + last_noise['vx']
        nx = state.X + last_noise['x']
        ny = state.Y + last_noise['y']
        nphi = state.phi + last_noise['phi']

        perceived = VehicleState(X=nx, Y=ny, phi=nphi, vx=nvx, vy=state.vy, omega=state.omega)

        # FIX UNPACKING: Usiamo *_ per essere pronti a ricevere qualsiasi numero di output
        e, theta_e, *extra = estimator.get_errors(perceived)

        # Recuperiamo l'indice (che dovrebbe essere l'ultimo o il quarto valore)
        idx = extra[-1] if extra else 0

        kappa = estimator.get_curvature(idx, lookahead=15)
        target_speed = v_max / (1 + sensibilita_curvatura * kappa)
        target_speed = np.clip(target_speed, v_min, v_max)

        kp, kd, mode = supervisor.update_and_get_gains(perceived.vx, use_filter=usa_filtro_supervisore)

        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speed, state.vx)

        u_in = VehicleInput(d=d_cmd, delta=delta_cmd)
        state = integrator.RK4(state, u_in)

        history['x'].append(state.X)
        history['y'].append(state.Y)
        history['vx'].append(state.vx)
        history['e'].append(e)
        history['theta_e'].append(theta_e)
        history['mode'].append(mode)
        history['target_v'].append(target_speed)

        if live_plot and i % 100 == 0:
            line_follower.set_data(history['x'], history['y'])
            current_pos.set_data([state.X], [state.Y])
            sensor_pos.set_data([perceived.X], [perceived.Y])
            ax.set_xlim(state.X - 15, state.X + 15)
            ax.set_ylim(state.Y - 15, state.Y + 15)
            plt.draw()
            plt.pause(0.001)

        if abs(e) > 10.0:
            print(f"--- SBANDAMENTO CRITICO a t={t:.2f}s ---")
            break

    # --- FINE SIMULAZIONE ---
    print("Salvataggio dati in corso...")

    # 1. Chiudiamo la modalità interattiva ma NON la finestra se vogliamo vederla
    if live_plot:
        plt.ioff()

        # 2. Calcolo RMSE
    actual_e = np.array(history['e'])
    rmse = np.sqrt(np.mean(actual_e ** 2)) if len(actual_e) > 0 else 0
    print(f"Simulazione terminata. RMSE: {rmse:.4f}m")

    # 3. SALVATAGGIO FISICO (Verifica che la cartella run_dir esista)
    save_simulation_data(run_dir, history)
    save_metadata(run_dir,
                  {"Test": "Sbandamento_Adattativo", "Noise": noise_magnitude},
                  {"RMSE": f"{rmse:.5f}m", "Status": "Completed" if abs(e) <= 10.0 else "Failed"})

    # 4. DASHBOARD FINALE
    # Questa funzione di solito crea una nuova figura.
    plot_dashboard(run_dir, history, path_points, f"{track_name} - Analisi Robustezza")

    print(f"Tutti i risultati sono stati salvati in: {run_dir}")

    # 5. BLOCCA LO SCHERMO per vedere i grafici
    plt.show()


if __name__ == "__main__":
    run_robustness_simulation('racing', usa_rk4=True, live_plot=True)