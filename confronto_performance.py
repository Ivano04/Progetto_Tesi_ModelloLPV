import numpy as np
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata, plot_comparison_results
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_simulation(track_type, adaptive=True):
    # Setup parametri
    dt = 0.001
    total_time = 25.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(track_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    supervisor = SupervisorS()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # Profilo di velocità crescente (per mettere in crisi il controllo fisso)
    target_speeds = np.linspace(0.5, 3.5, steps)
    history_e = []

    # Guadagni fissi (usiamo quelli del regime LOW per vedere come peggiorano ad alta velocità)
    fixed_kp, fixed_kd = -1.2, -0.2

    for i in range(steps):
        e, theta_e, _ = estimator.get_errors(state)

        if adaptive:
            kp, kd, _ = supervisor.update_and_get_gains(state.vx)
        else:
            kp, kd = fixed_kp, fixed_kd

        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speeds[i], state.vx)

        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()
        state = integrator.Eulero(state, u_in)
        history_e.append(e)

    return np.array(history_e), track_name


def main_confronto():
    # Scegliamo la pista racing perché è la più completa
    scenario = 'racing'

    print(f"1/2 Esecuzione simulazione ADATTATIVA LPV...")
    error_adaptive, track_name = run_simulation(scenario, adaptive=True)

    print(f"2/2 Esecuzione simulazione con GUADAGNI FISSI...")
    error_fixed, _ = run_simulation(scenario, adaptive=False)

    # Setup cartella risultati specifica
    run_dir = setup_results_dir(track_name, "Confronto_Performance")

    # Calcolo metriche di performance
    rmse_adapt = np.sqrt(np.mean(error_adaptive ** 2))
    rmse_fixed = np.sqrt(np.mean(error_fixed ** 2))
    miglioramento = ((rmse_fixed - rmse_adapt) / rmse_fixed) * 100

    # Salvataggio Metadati e CSV
    stats = {
        "RMSE_LPV": f"{rmse_adapt:.5f} m",
        "RMSE_Fisso": f"{rmse_fixed:.5f} m",
        "Miglioramento_Percentuale": f"{miglioramento:.2f}%"
    }
    save_metadata(run_dir, {"Pista": track_name, "Metodo": "LPV vs Fixed (LOW)"}, stats)

    # Salvataggio dati CSV (per grafici su Excel)
    save_simulation_data(run_dir, {"errore_lpv": error_adaptive, "errore_fisso": error_fixed})

    print("\n" + "=" * 30)
    print("   RISULTATI CONFRONTO")
    print("=" * 30)
    print(f"RMSE LPV:   {rmse_adapt:.5f} m")
    print(f"RMSE FISSO: {rmse_fixed:.5f} m")
    print(f"MIGLIORAMENTO: {miglioramento:.1f}%")
    print("=" * 30)

    # Plot e visualizzazione a schermo
    plot_comparison_results(run_dir, error_adaptive, error_fixed, track_name)


if __name__ == "__main__":
    main_confronto()