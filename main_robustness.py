import numpy as np
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator, Generator_Noise
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata, plot_dashboard
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_robustness_simulation(scenario_key):
    dt = 0.001
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)

    # Cartella: Grafici / <Traiettoria> / Test_Robustezza
    run_dir = setup_results_dir(track_name, "Test_Robustezza")

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # Rumore sui sensori (velocità e posizione)
    noise_gen = Generator_Noise(disturb_vx=True, disturb_position=True, magnitude=0.07)

    target_speeds = np.linspace(0.5, 3.0, steps)
    history = {'x': [], 'y': [], 'vx': [], 'e': [], 'theta_e': [], 'mode': []}

    print(f"RUN ROBUSTEZZA: {track_name} (Noise Magnitude: 0.07)")
    for i in range(steps):
        t = i * dt
        # Stato Percezione (Sporco)
        nvx = state.vx + noise_gen.get_disturbance(t, 'vx')
        nx = state.X + noise_gen.get_disturbance(t, 'position')
        ny = state.Y + noise_gen.get_disturbance(t, 'position')
        perceived = VehicleState(X=nx, Y=ny, phi=state.phi, vx=nvx, vy=state.vy, omega=state.omega)

        e, theta_e, _ = estimator.get_errors(perceived)
        kp, kd, mode = supervisor.update_and_get_gains(perceived.vx)
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)

        # Fisica Reale
        d_cmd = longitudinal_ctrl.compute(target_speeds[i], state.vx)
        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()
        state = integrator.Eulero(state, u_in)

        history['x'].append(state.X);
        history['y'].append(state.Y)
        history['vx'].append(state.vx);
        history['e'].append(e)
        history['theta_e'].append(theta_e);
        history['mode'].append(mode)

    # Salvataggio e Plot a schermo
    rmse = np.sqrt(np.mean(np.array(history['e']) ** 2))
    save_simulation_data(run_dir, history)
    save_metadata(run_dir, {"Test": "Robustezza", "Noise_Mag": 0.07}, {"RMSE_Laterale": f"{rmse:.5f}m"})

    plot_dashboard(run_dir, history, path_points, f"({track_name} - Robustezza)")


if __name__ == "__main__":
    # Testiamo i casi più probanti per la robustezza
    for scenario in ['racing', 'eight']:
        run_robustness_simulation(scenario)
    print("\n✓ ANALISI DI ROBUSTEZZA COMPLETATA.")