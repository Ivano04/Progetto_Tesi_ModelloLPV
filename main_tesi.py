import numpy as np
import matplotlib.pyplot as plt
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata, plot_dashboard
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_nominal_simulation(scenario_key, usa_rk4=True, live_plot=True):
    # --- CONFIGURAZIONE PARAMETRI ---
    integrator_type = "RK4" if usa_rk4 else "Eulero"
    dt = 0.001
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)
    path_array = np.array(path_points)

    run_dir = setup_results_dir(track_name, "Analisi_Nominale", integrator_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    target_speeds = np.linspace(0.5, 3.0, steps)
    history = {'x': [], 'y': [], 'vx': [], 'e': [], 'theta_e': [], 'mode': []}

    # SETUP VISUALIZZAZIONE LIVE LEGGERA ---
    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(path_array[:, 0], path_array[:, 1], 'k--', alpha=0.2)
        line_follower, = ax.plot([], [], 'b-', linewidth=1.5, label='Scia')
        current_pos, = ax.plot([], [], 'ro', markersize=6, label='Auto')
        ax.set_title(f"Live: {track_name} ({integrator_type})")
        ax.grid(True, alpha=0.2)
    # ------------------------------------------

    print(f"AVVIO SIMULAZIONE: {track_name} con {integrator_type}...")

    for i in range(steps):
        e, theta_e, _ = estimator.get_errors(state)
        kp, kd, mode = supervisor.update_and_get_gains(state.vx)
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speeds[i], state.vx)
        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()

        if usa_rk4:
            state = integrator.RK4(state, u_in)
        else:
            state = integrator.Eulero(state, u_in)

        history['x'].append(state.X)
        history['y'].append(state.Y)
        history['vx'].append(state.vx)
        history['e'].append(e)
        history['theta_e'].append(theta_e)
        history['mode'].append(mode)

        if live_plot and i % 250 == 0:  # Ogni 0.25 secondi simulati
            line_follower.set_data(history['x'], history['y'])
            current_pos.set_data([state.X], [state.Y])

            # Zoom dinamico senza forzare l'aspect ratio fisso (evita il crash)
            ax.set_xlim(state.X - 8, state.X + 8)
            ax.set_ylim(state.Y - 8, state.Y + 8)

            plt.pause(0.001)

    if live_plot:
        plt.ioff()
        plt.close(fig)

    save_simulation_data(run_dir, history)
    rmse = np.sqrt(np.mean(np.array(history['e']) ** 2))
    save_metadata(run_dir,
                  {"Stato": "Nominale", "Track": track_name, "Integratore": integrator_type},
                  {"RMSE": f"{rmse:.5f}m"})
    plot_dashboard(run_dir, history, path_points, f"{track_name} - {integrator_type}")


if __name__ == "__main__":
    # SCEGLI QUI IL CIRCUITO DA TESTARE: 'racing', 'circular', 'eight'
    circuito_scelto = 'racing'

    run_nominal_simulation(circuito_scelto, usa_rk4=False, live_plot=True)

    print(f"\n✓ Simulazione su {circuito_scelto} completata.")