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
    dt = 0.01
    steps = int(20.0 / dt)
    path_points, _ = get_trajectory(track_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

    # Inizializzazione stato
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=22.5, vy=0.0, omega=0.0)

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

        # 1. FIX UNPACKING: Usiamo *_ per gestire eventuali valori extra (curvatura, index, ecc.)
        e, theta_e, *_ = estimator.get_errors(perceived)

        # Limite di sicurezza
        if abs(e) > 1.5 or np.isnan(e):
            success = False
            break

        kp, kd, _ = supervisor.update_and_get_gains(perceived.vx, use_filter=True)
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)

        # 2. FIX KEYWORD: Ripristinato 'd' invece di 'a' come richiesto dalla tua classe VehicleInput
        u_in = VehicleInput(d=0.5, delta=delta_cmd).saturate()

        state = integrator.RK4(state, u_in)
        history_e.append(e)

    rmse = np.sqrt(np.mean(np.array(history_e) ** 2)) if (len(history_e) > 0 and success) else 2.0
    return rmse, success


def main_stress_test():
    noise_levels = np.arange(0.02, 0.42, 0.04)
    results_rmse = []
    results_success = []

    print("=== AVVIO NOISE STRESS TEST (Mugello + RK4 + Filtro) ===")

    for mag in noise_levels:
        rmse, success = run_single_stress_test(mag)
        results_rmse.append(rmse)
        results_success.append(success)
        status = "OK" if success else "FALLITO"
        print(f"Mag: {mag:.2f} | Status: {status} | RMSE: {rmse:.4f}m")

    # Setup directory e salvataggio
    run_dir = setup_results_dir("Circuito_Mugello", "Analisi_Stress_Test", "RK4")

    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, results_rmse, 'b-o', linewidth=2, label='RMSE Laterale')

    if False in results_success:
        break_idx = np.where(np.array(results_success) == False)[0][0]
        plt.axvline(x=noise_levels[break_idx], color='r', linestyle='--', label='Punto di Rottura')
        break_val = f"{noise_levels[break_idx]:.2f}"
    else:
        break_val = "Non raggiunto"

    plt.title("Analisi di Sensibilità: Rumore vs Precisione", fontweight='bold')
    plt.xlabel("Magnitudo Rumore (m - m/s)")
    plt.ylabel("RMSE Errore Laterale [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.savefig(os.path.join(run_dir, "Curva_Sensibilita_Rumore.png"), dpi=300)
    save_metadata(run_dir, {"Test": "Stress Test Rumore", "Initial_Vx": "10.0"}, {"Punto_Rottura": break_val})
    plt.show()


if __name__ == "__main__":
    main_stress_test()