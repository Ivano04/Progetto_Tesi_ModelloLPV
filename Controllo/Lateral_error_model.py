import numpy as np
import math


class LateralErrorEstimator:
    def __init__(self, path):
        # path è la lista di tuple (x, y) da trajectory_generator.py
        self.path = np.array(path)

    def get_errors(self, vehicle_state):
        # 1. Trova l'indice del punto più vicino sulla traiettoria
        dists = np.linalg.norm(self.path - np.array([vehicle_state.X, vehicle_state.Y]), axis=1)
        idx = np.argmin(dists)

        nearest_point = self.path[idx]

        # 2. Calcola l'angolo della tangente (phi_path) nel punto più vicino
        # Usiamo il punto successivo per stimare la pendenza
        next_idx = (idx + 1) % len(self.path)
        p1 = self.path[idx]
        p2 = self.path[next_idx]
        phi_path = math.atan2(p2[1] - p1[1], p2[0] - p1[0])

        # 3. Calcolo Errore Laterale (e) con segno
        # e > 0 se il veicolo è a sinistra del percorso, e < 0 se a destra
        dx = vehicle_state.X - nearest_point[0]
        dy = vehicle_state.Y - nearest_point[1]

        # Proiezione trasversale (Frenet frame)
        error_e = -dx * math.sin(phi_path) + dy * math.cos(phi_path)

        # 4. Calcolo Errore di Orientamento (theta_e)
        # Deve essere normalizzato tra -pi e pi
        error_theta = vehicle_state.phi - phi_path

        # Normalizzazione (usando la funzione già presente nel modello del collega)
        from Veicolo.Vehicle_model import normalize_angle
        error_theta = normalize_angle(error_theta)

        return error_e, error_theta, phi_path