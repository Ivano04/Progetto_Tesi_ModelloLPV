import numpy as np
import matplotlib.pyplot as plt
import os
from Veicolo.Vehicle_model import DynamicBicycleModel, VehicleState, VehicleInput, VehicleIntegrator, Generator_Noise
from Utils.trajectory_generator import get_trajectory
from Utils.plotting_utils import setup_results_dir, save_metadata
from Controllo.Supervisor_S import SupervisorS
from Controllo.Lateral_error_model import LateralErrorEstimator
from Controllo.PD_controller import LateralPDController


def run_single_stress_test(noise_magnitude, track_type='racing'):
    dt = 0.001
    steps = int(25.0 / dt)
    path_points, _ = get_trajectory(track_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=0.1, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()
    noise_gen = Generator_Noise(disturb_vx=True, disturb_position=True, magnitude=noise_magnitude)

    history_e = []
    success = True

    for i in range(steps):
        t = i * dt
        # Percezione rumorosa
        nvx = state.vx + noise_gen.get_disturbance(t, 'vx')
        nx = state.X + noise_gen.get_disturbance(t, 'position')
        ny = state.Y + noise_gen.get_disturbance(t, 'position')
        perceived = VehicleState(X=nx, Y=ny, phi=state.phi, vx=nvx, vy=state.vy, omega=state.omega)

        e, theta_e, _ = estimator.get_errors(perceived)

        # Limite di sicurezza: se l'errore supera 1 metro, consideriamo il test fallito
        if abs(e) > 1.0:
            success = False
            break

        kp, kd, _ = supervisor.update_and_get_gains(perceived.vx, use_filter=True)
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)
        u_in = VehicleInput(d=0.5, delta=delta_cmd).saturate()  # Velocità costante per isolare il rumore

        state = integrator.RK4(state, u_in)
        history_e.append(e)

    rmse = np.sqrt(np.mean(np.array(history_e) ** 2)) if len(history_e) > 0 else float('nan')
    return rmse, success


def main_stress_test():
    # Range di rumore da testare: da 0.02 a 0.40 con passi di 0.02
    noise_levels = np.arange(0.02, 0.42, 0.02)
    results_rmse = []
    results_success = []

    print("=== AVVIO NOISE STRESS TEST (Mugello + RK4 + Filtro) ===")

    for mag in noise_levels:
        print(f"Testing Magnitude: {mag:.2f}...", end=" ")
        rmse, success = run_single_stress_test(mag)
        results_rmse.append(rmse)
        results_success.append(success)
        status = "OK" if success else "FALLITO"
        print(f"Result: {status} | RMSE: {rmse:.5f}m")

    # Creazione cartella risultati
    run_dir = setup_results_dir("Circuito_Mugello", "Analisi_Stress_Test", "RK4")

    # Grafico della Sensibilità
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, results_rmse, 'b-o', linewidth=2, label='RMSE Laterale')
    plt.axvline(x=noise_levels[np.where(np.array(results_success) == False)[0][0]] if False in results_success else 0.4,
                color='r', linestyle='--', label='Punto di Rottura (Divergenza)')

    plt.title("Analisi di Sensibilità: Rumore vs Precisione (Mugello)", fontweight='bold')
    plt.xlabel("Magnitudo Rumore (m - m/s)")
    plt.ylabel("RMSE Errore Laterale [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(os.path.join(run_dir, "Curva_Sensibilita_Rumore.png"), dpi=300)
    plt.show()

    save_metadata(run_dir, {"Test": "Stress Test Rumore", "Range": "0.02 - 0.40"},
                  {
                      "Punto_Rottura": f"{noise_levels[np.where(np.array(results_success) == False)[0][0]]:.2f}" if False in results_success else "Non raggiunto"})


if __name__ == "__main__":
    main_stress_test()