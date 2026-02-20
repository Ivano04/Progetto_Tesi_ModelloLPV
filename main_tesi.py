import numpy as np
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata, plot_dashboard
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_nominal_simulation(scenario_key):
    dt = 0.001
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)

    # Cartella: Grafici / <Traiettoria> / Analisi_Nominale
    run_dir = setup_results_dir(track_name, "Analisi_Nominale")

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    target_speeds = np.linspace(0.5, 3.0, steps)
    history = {'x': [], 'y': [], 'vx': [], 'e': [], 'theta_e': [], 'mode': []}

    print(f"RUN NOMINALE: {track_name}")
    for i in range(steps):
        e, theta_e, _ = estimator.get_errors(state)
        kp, kd, mode = supervisor.update_and_get_gains(state.vx)
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speeds[i], state.vx)

        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()
        state = integrator.Eulero(state, u_in)

        history['x'].append(state.X);
        history['y'].append(state.Y)
        history['vx'].append(state.vx);
        history['e'].append(e)
        history['theta_e'].append(theta_e);
        history['mode'].append(mode)

    # Salvataggio dati e plot a schermo
    save_simulation_data(run_dir, history)
    rmse = np.sqrt(np.mean(np.array(history['e']) ** 2))
    save_metadata(run_dir, {"Stato": "Nominale", "Track": track_name}, {"RMSE": f"{rmse:.5f}m"})

    plot_dashboard(run_dir, history, path_points, f"({track_name} - Nominale)")


if __name__ == "__main__":
    # Testiamo tutte le traiettorie implementate nel generatore
    for scenario in ['racing', 'circular', 'eight']:
        run_nominal_simulation(scenario)
    print("\n✓ ANALISI NOMINALE COMPLETATA CON SUCCESSO.")