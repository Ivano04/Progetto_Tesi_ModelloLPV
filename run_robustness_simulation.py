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
    """
    Esegue un test di sbandamento e robustezza specifico per valutare
    le transizioni tra i 4 livelli del supervisore LPV.
    """
    # --- CONFIGURAZIONE ---
    usa_filtro_supervisore = True  # Fondamentale con dati rumorosi per evitare saltellamenti tra i 4 stati
    integrator_type = "RK4" if usa_rk4 else "Eulero"

    dt = 0.005  # Passo temporale bilanciato tra velocità e precisione
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)
    path_array = np.array(path_points)

    # Parametri Velocità Adattativa
    v_max = 3.5
    v_min = 1.0
    sensibilita_curvatura = 12.0

    # Directory dei risultati specifica per il test a 4 livelli
    run_dir = setup_results_dir(track_name, f"Test_Sbandamento_LPV_4Livelli", integrator_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # --- CONFIGURAZIONE RUMORE ---
    # Impostato a 0.1 per un test di sbandamento significativo ma controllabile
    noise_magnitude = 0.1
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
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(path_array[:, 0], path_array[:, 1], 'k--', alpha=0.15)
        line_follower, = ax.plot([], [], 'b-', linewidth=1.5, label='Traiettoria Reale')
        current_pos, = ax.plot([], [], 'ro', markersize=6)
        sensor_pos, = ax.plot([], [], 'gx', markersize=4, alpha=0.3, label='Percezione Rumorosa')
        ax.legend()

    print(f"AVVIO TEST SBANDAMENTO (4 LIVELLI): {track_name} | Rumore: {noise_magnitude}")

    for i in range(steps):
        t = i * dt

        # Simuliamo un campionamento sensori più lento (ogni 10 step) per il rumore
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

        # 1. Calcolo Errori e Curvatura
        e, theta_e, *extra = estimator.get_errors(perceived)
        idx = extra[-1] if extra else 0

        kappa = estimator.get_curvature(idx, lookahead=15)
        target_speed = v_max / (1 + sensibilita_curvatura * kappa)
        target_speed = np.clip(target_speed, v_min, v_max)

        # 2. Supervisore LPV (Utilizza le nuove soglie e i nuovi guadagni)
        kp, kd, mode = supervisor.update_and_get_gains(perceived.vx, use_filter=usa_filtro_supervisore)

        # 3. Leggi di Controllo
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speed, state.vx)

        # 4. Integrazione Fisica Reale
        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()
        state = integrator.RK4(state, u_in)

        # Registrazione dati REALI
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

        # Se l'errore è troppo alto, il veicolo è sbandato irrimediabilmente
        if abs(e) > 8.0:
            print(f"--- SBANDAMENTO CRITICO a t={t:.2f}s ---")
            break

    # --- FINE SIMULAZIONE ---
    if live_plot:
        plt.ioff()

    actual_e = np.array(history['e'])
    rmse = np.sqrt(np.mean(actual_e ** 2)) if len(actual_e) > 0 else 0
    print(f"Simulazione terminata. RMSE: {rmse:.4f}m")

    # Salvataggio Dati e Metadati
    save_simulation_data(run_dir, history)
    save_metadata(run_dir,
                  {"Test": "Sbandamento_LPV_4Livelli", "Noise_Mag": noise_magnitude, "V_max": v_max},
                  {"RMSE": f"{rmse:.5f}m", "Status": "Completed" if abs(e) <= 8.0 else "Failed"})

    # Dashboard Finale (aggiornata con le 4 zone)
    plot_dashboard(run_dir, history, path_points, f"{track_name} - Sbandamento 4 Livelli")

    print(f"Risultati salvati in: {run_dir}")
    plt.show()


if __name__ == "__main__":
    # Test sul circuito racing (Mugello) per vedere il comportamento sotto stress
    run_robustness_simulation('racing', usa_rk4=True, live_plot=True)