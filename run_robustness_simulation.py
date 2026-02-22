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
    # --- CONFIGURAZIONE ---
    usa_filtro_supervisore = False  # Disattiviamo il filtro per vedere lo sbandamento puro
    integrator_type = "RK4" if usa_rk4 else "Eulero"

    dt = 0.001
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)
    path_array = np.array(path_points)

    run_dir = setup_results_dir(track_name, f"Test_Sbandamento_Intensita", integrator_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # --- CONFIGURAZIONE RUMORE STILE FASE 1 ---
    noise_magnitude = 0.5  # Prova a salire da 0.07 a 0.5, 1.5, 4.07
    noise_gen = Generator_Noise(
        disturb_vx=True,
        disturb_position=True,
        disturb_heading=True,  # Fondamentale per lo sbandamento
        magnitude=noise_magnitude,
        magnitude_position=noise_magnitude,
        magnitude_heading=noise_magnitude * 2.97,  # Rumore sull'angolo [rad]
        disturbance_type="noise"
    )

    target_speeds = np.linspace(0.5, 3.0, steps)
    history = {'x': [], 'y': [], 'vx': [], 'e': [], 'theta_e': []}

    # Variabili per "persistenza" del rumore (aggiornamento ogni 10ms)
    last_noise = {'x': 0, 'y': 0, 'phi': 0, 'vx': 0}

    if live_plot:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.plot(path_array[:, 0], path_array[:, 1], 'k--', alpha=0.15)
        line_follower, = ax.plot([], [], 'b-', linewidth=1.5, label='Traiettoria Reale')
        current_pos, = ax.plot([], [], 'ro', markersize=6)
        # Visualizziamo anche dove il sensore "crede" di essere
        sensor_pos, = ax.plot([], [], 'gx', markersize=4, alpha=0.5, label='Percezione Rumorosa')
        ax.legend()

    for i in range(steps):
        t = i * dt

        # Aggiornamento rumore ogni 10 step (100Hz) per dare tempo alla fisica di reagire
        if i % 10 == 0:
            last_noise['vx'] = noise_gen.get_disturbance(t, 'vx')
            last_noise['x'] = noise_gen.get_disturbance(t, 'position')
            last_noise['y'] = noise_gen.get_disturbance(t, 'position')
            last_noise['phi'] = noise_gen.get_disturbance(t, 'heading')

        # 1. Percezione "Sporca" (Stile Fase 1)
        nvx = state.vx + last_noise['vx']
        nx = state.X + last_noise['x']
        ny = state.Y + last_noise['y']
        nphi = state.phi + last_noise['phi']  # Il controller ora riceve un angolo errato!

        perceived = VehicleState(X=nx, Y=ny, phi=nphi, vx=nvx, vy=state.vy, omega=state.omega)

        # 2. Controllo basato su dati rumorosi
        e, theta_e, _ = estimator.get_errors(perceived)
        kp, kd, mode = supervisor.update_and_get_gains(perceived.vx, use_filter=usa_filtro_supervisore)

        # 3. Leggi di Controllo (SENZA SATURAZIONE per massimizzare l'effetto)
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speeds[i], state.vx)

        # 4. Fisica Reale (Applichiamo il comando errato al veicolo vero)
        u_in = VehicleInput(d=d_cmd, delta=delta_cmd)  # Rimosso .saturate()

        state = integrator.RK4(state, u_in)

        history['x'].append(state.X);
        history['y'].append(state.Y);
        history['e'].append(e)

        if live_plot and i % 250 == 0:
            line_follower.set_data(history['x'], history['y'])
            current_pos.set_data([state.X], [state.Y])
            sensor_pos.set_data([perceived.X], [perceived.Y])
            ax.set_xlim(state.X - 10, state.X + 10);
            ax.set_ylim(state.Y - 10, state.Y + 10)
            plt.pause(0.001)

        # Se l'auto sbanda troppo (es. 10 metri fuori), interrompiamo
        if abs(e) > 10.0:
            print(f"--- SBANDAMENTO CRITICO a t={t:.2f}s ---")
            break

    if live_plot: plt.close(fig)
    print(f"Simulazione terminata con RMSE: {np.sqrt(np.mean(np.array(history['e']) ** 2)):.4f}m")


if __name__ == "__main__":
    run_robustness_simulation('circular', usa_rk4=True, live_plot=True)