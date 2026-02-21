import numpy as np
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata, plot_dashboard
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_nominal_simulation(scenario_key):
    # --- CONFIGURAZIONE INTEGRATORE ---
    # Cambiare in False per eseguire il test con il modello di Eulero
    usa_rk4 = False
    integrator_type = "RK4" if usa_rk4 else "Eulero"
    # ----------------------------------

    dt = 0.001
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)

    # La cartella viene creata automaticamente in Grafici_RK4/ o Grafici_Eulero/
    run_dir = setup_results_dir(track_name, "Analisi_Nominale", integrator_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

    # Inizializzazione stato: parte dal primo punto della traiettoria
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # Profilo di velocità target crescente
    target_speeds = np.linspace(0.5, 3.0, steps)
    history = {'x': [], 'y': [], 'vx': [], 'e': [], 'theta_e': [], 'mode': []}

    print(f"RUN NOMINALE ({integrator_type}): {track_name}")

    for i in range(steps):
        # 1. Stima degli errori di inseguimento
        e, theta_e, _ = estimator.get_errors(state)

        # 2. Gain Scheduling tramite Supervisore LPV
        kp, kd, mode = supervisor.update_and_get_gains(state.vx)

        # 3. Calcolo dei comandi di controllo (Laterale e Longitudinale)
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speeds[i], state.vx)

        # 4. Saturazione e applicazione dell'input al modello
        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()

        # 5. Integrazione dello stato (Scelta automatica del metodo)
        if usa_rk4:
            state = integrator.RK4(state, u_in)
        else:
            state = integrator.Eulero(state, u_in)

        # Registrazione dati per analisi post-processamento
        history['x'].append(state.X)
        history['y'].append(state.Y)
        history['vx'].append(state.vx)
        history['e'].append(e)
        history['theta_e'].append(theta_e)
        history['mode'].append(mode)

    # --- SALVATAGGIO RISULTATI ---
    save_simulation_data(run_dir, history)

    # Calcolo dell'errore quadratico medio (RMSE)
    rmse = np.sqrt(np.mean(np.array(history['e']) ** 2))

    # Salvataggio report testuale
    save_metadata(
        run_dir,
        {"Stato": "Nominale", "Track": track_name, "Integratore": integrator_type},
        {"RMSE_Laterale": f"{rmse:.5f}m"}
    )

    # Generazione Dashboard Grafica
    plot_dashboard(run_dir, history, path_points, f"({track_name} - Nominale - {integrator_type})")


if __name__ == "__main__":
    # Esecuzione della simulazione nominale su tutte le traiettorie disponibili
    scenari = ['racing', 'circular', 'eight']

    for scenario in scenari:
        run_nominal_simulation(scenario)

    print(f"\n✓ ANALISI NOMINALE COMPLETATA CON SUCCESSO.")