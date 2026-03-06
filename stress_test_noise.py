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
    """
    Esegue un singolo test di tenuta con un livello di rumore specifico.
    Valuta se il sistema LPV a 4 livelli riesce a mantenere il veicolo in pista.
    """
    dt = 0.01
    total_sim_time = 20.0
    steps = int(total_sim_time / dt)
    path_points, _ = get_trajectory(track_type)

    model = DynamicBicycleModel()
    integrator = VehicleIntegrator(model, dt=dt)

    # Inizializzazione: vx=2.5 m/s per testare la stabilità nei regimi alti (HIGH mode)
    initial_vx = 2.5
    state = VehicleState(X=path_points[0][0], Y=path_points[0][1], phi=0.0, vx=initial_vx, vy=0.0, omega=0.0)

    supervisor = SupervisorS()
    estimator = LateralErrorEstimator(path_points)
    lateral_ctrl = LateralPDController()

    # Generatore di rumore gaussiano su velocità e posizione
    noise_gen = Generator_Noise(disturb_vx=True, disturb_position=True, magnitude=noise_magnitude)

    history_e = []
    success = True

    for i in range(steps):
        t = i * dt

        # 1. Percezione rumorosa
        nvx = state.vx + noise_gen.get_disturbance(t, 'vx')
        nx = state.X + noise_gen.get_disturbance(t, 'position')
        ny = state.Y + noise_gen.get_disturbance(t, 'position')
        perceived = VehicleState(X=nx, Y=ny, phi=state.phi, vx=nvx, vy=state.vy, omega=state.omega)

        # 2. Calcolo Errori (Unpacking flessibile per 4 livelli)
        e, theta_e, *_ = estimator.get_errors(perceived)

        # Limite di sicurezza: se l'errore supera 1.5m, il test è considerato FALLITO
        if abs(e) > 1.5 or np.isnan(e):
            success = False
            break

        # 3. Controllo LPV con Filtro attivo (essenziale per lo stress test)
        kp, kd, _ = supervisor.update_and_get_gains(perceived.vx, use_filter=True)
        delta_cmd = lateral_ctrl.compute_control(e, theta_e, kp, kd)

        # 4. Input e Integrazione (d=0.5 mantiene la velocità costante/leggera accelerazione)
        u_in = VehicleInput(d=0.5, delta=delta_cmd).saturate()

        state = integrator.RK4(state, u_in)
        history_e.append(e)

    # Calcolo RMSE: se fallisce, assegniamo un errore convenzionale alto (2.0m)
    rmse = np.sqrt(np.mean(np.array(history_e) ** 2)) if (len(history_e) > 0 and success) else 2.0
    return rmse, success


def main_stress_test():
    # Range di rumore da testare: da quasi nullo (0.02) a molto pesante (0.42)
    noise_levels = np.arange(0.02, 0.42, 0.04)
    results_rmse = []
    results_success = []

    print("=== AVVIO NOISE STRESS TEST: LPV 4 LIVELLI (Mugello + RK4) ===")

    for mag in noise_levels:
        rmse, success = run_single_stress_test(mag)
        results_rmse.append(rmse)
        results_success.append(success)
        status = "OK" if success else "FALLITO"
        print(f"Magnitudo Rumore: {mag:.2f} | Status: {status} | RMSE: {rmse:.4f}m")

    # Creazione directory risultati
    run_dir = setup_results_dir("Circuito_Mugello", "Analisi_Stress_Test_4Livelli", "RK4")

    # Generazione grafico di sensibilità
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, results_rmse, 'b-o', linewidth=2, label='RMSE Laterale (4 Livelli)')

    # Identificazione punto di rottura
    if False in results_success:
        break_idx = np.where(np.array(results_success) == False)[0][0]
        plt.axvline(x=noise_levels[break_idx], color='r', linestyle='--', label='Punto di Rottura')
        break_val = f"{noise_levels[break_idx]:.2f}"
    else:
        break_val = "Non raggiunto (Sistema Molto Robusto)"

    plt.title("Analisi di Sensibilità LPV: Rumore vs Precisione", fontweight='bold')
    plt.xlabel("Magnitudo Rumore (m - m/s)")
    plt.ylabel("RMSE Errore Laterale [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Salvataggio
    plt.savefig(os.path.join(run_dir, "Curva_Sensibilita_Rumore.png"), dpi=300)
    save_metadata(run_dir,
                  {"Test": "Stress Test Rumore LPV", "Initial_Vx": "2.5", "Soglie": "4 Livelli"},
                  {"Punto_Rottura": break_val})

    print(f"\n✓ Stress test completato. Punto di rottura: {break_val}")
    plt.show()


if __name__ == "__main__":
    main_stress_test()