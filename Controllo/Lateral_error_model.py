import numpy as np
import math


class LateralErrorEstimator:
    def __init__(self, path):
        # path è la lista di tuple (x, y) da trajectory_generator.py
        self.path = np.array(path)

    def get_errors(self, vehicle_state):
        """
        Calcola gli errori di tracking e restituisce anche l'indice del punto
        più vicino per permettere il calcolo della curvatura locale.
        """
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
        error_theta = vehicle_state.phi - phi_path

        # Normalizzazione (usando la funzione presente nel modello del veicolo)
        from Veicolo.Vehicle_model import normalize_angle
        error_theta = normalize_angle(error_theta)

        # Restituiamo anche idx per il calcolo della curvatura nel main
        return error_e, error_theta, phi_path, idx

    def get_curvature(self, idx, lookahead=10):
        """
        Calcola la curvatura locale (kappa) guardando la variazione dell'angolo
        tra il punto corrente e un punto 'lookahead' più avanti.
        """
        # Indici dei punti da confrontare
        p1_idx = idx
        p2_idx = (idx + lookahead) % len(self.path)

        # Angolo nel punto corrente
        p1 = self.path[p1_idx]
        p1_next = self.path[(p1_idx + 1) % len(self.path)]
        phi1 = math.atan2(p1_next[1] - p1[1], p1_next[0] - p1[0])

        # Angolo nel punto lookahead
        p2 = self.path[p2_idx]
        p2_next = self.path[(p2_idx + 1) % len(self.path)]
        phi2 = math.atan2(p2_next[1] - p2[1], p2_next[0] - p2[0])

        # Variazione angolare normalizzata
        from Veicolo.Vehicle_model import normalize_angle
        d_phi = abs(normalize_angle(phi2 - phi1))

        # Distanza spaziale tra i due punti
        dist = np.linalg.norm(self.path[p2_idx] - self.path[p1_idx])

        # Curvatura approssimata: variazione angolo / variazione spazio
        curvature = d_phi / (dist + 1e-5)

        return curvature