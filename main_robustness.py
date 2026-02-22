import numpy as np
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator, Generator_Noise
from Veicolo.PID_controller import VelocityPIDController
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_simulation_data, save_metadata, plot_dashboard
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_robustness_simulation(scenario_key):
    # --- CONFIGURAZIONE INTEGRATORE ---

    #impostare su True per attivare il filtro ed eliminare il chattering anche in caso di rumore forte
    usa_filtro_supervisore = True
    supervisor_status = "Filtrato" if usa_filtro_supervisore else "Standard"

    # Impostare su False per verificare il fallimento numerico di Eulero con rumore 0.07
    usa_rk4 = True
    integrator_type = "RK4" if usa_rk4 else "Eulero"


    dt = 0.001
    total_time = 30.0
    steps = int(total_time / dt)
    path_points, track_name = get_trajectory(scenario_key)

    # La cartella viene creata in Grafici_RK4/ o Grafici_Eulero/ in base all'integratore scelto
    run_dir = setup_results_dir(track_name, "Test_Robustezza", integrator_type)

    test_type = f"Test_Robustezza_{supervisor_status}"
    run_dir = setup_results_dir(track_name, test_type, integrator_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

    # Inizializzazione: posizione di partenza della traiettoria prescelta
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    longitudinal_ctrl = VelocityPIDController(kp=1.0, ki=0.1, dt=dt)

    # Rumore sui sensori (velocità e posizione) per testare la robustezza del controllo e dell'isteresi
    noise_magnitude = 0.07
    noise_gen = Generator_Noise(disturb_vx=True, disturb_position=True, magnitude=noise_magnitude)

    target_speeds = np.linspace(0.5, 3.0, steps)
    history = {'x': [], 'y': [], 'vx': [], 'e': [], 'theta_e': [], 'mode': []}

    print(f"RUN ROBUSTEZZA ({integrator_type}): {track_name} (Rumore: {noise_magnitude})")

    for i in range(steps):
        t = i * dt

        # 1. Simulazione Errore Sensoriale: Stato Percezione (Sporco)
        nvx = state.vx + noise_gen.get_disturbance(t, 'vx')
        nx = state.X + noise_gen.get_disturbance(t, 'position')
        ny = state.Y + noise_gen.get_disturbance(t, 'position')

        # Il controllore vede lo stato "perceived", non quello reale
        perceived = VehicleState(X=nx, Y=ny, phi=state.phi, vx=nvx, vy=state.vy, omega=state.omega)

        # 2. Calcolo errori e guadagni basati sui dati rumorosi
        e, theta_e, _ = estimator.get_errors(perceived)
        kp, kd, mode = supervisor.update_and_get_gains(perceived.vx, use_filter = usa_filtro_supervisore)

        # 3. Leggi di Controllo
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        d_cmd = longitudinal_ctrl.compute(target_speeds[i], state.vx)

        # 4. Applicazione al veicolo (Fisica Reale)
        u_in = VehicleInput(d=d_cmd, delta=delta_cmd).saturate()

        # Integrazione dello stato reale
        if usa_rk4:
            state = integrator.RK4(state, u_in)
        else:
            state = integrator.Eulero(state, u_in)

        # Registrazione dati
        history['x'].append(state.X)
        history['y'].append(state.Y)
        history['vx'].append(state.vx)
        history['e'].append(e)
        history['theta_e'].append(theta_e)
        history['mode'].append(mode)

    # --- SALVATAGGIO E REPORTISTICA ---
    save_simulation_data(run_dir, history)

    # Calcolo RMSE dell'errore laterale in presenza di rumore
    rmse = np.sqrt(np.mean(np.array(history['e']) ** 2))

    save_metadata(
        run_dir,
        {
            "Test": "Robustezza",
            "Noise_Mag": noise_magnitude,
            "Integratore": integrator_type,
            "Track": track_name
        },
        {"RMSE_Laterale": f"{rmse:.5f}m"}
    )

    # Dashboard grafica con suffisso dell'integratore
    plot_dashboard(run_dir, history, path_points, f"({track_name} - Robustezza - {integrator_type})")


if __name__ == "__main__":
    # Testiamo la robustezza su tutti gli scenari implementati
    scenari = ['racing', 'eight', 'circular']

    for scenario in scenari:
        run_robustness_simulation(scenario)

    print("\n✓ ANALISI DI ROBUSTEZZA COMPLETATA.")