import numpy as np


class LateralPDController:
    def __init__(self, steering_limit: float = 0.35):
        """
        Inizializza il controllore PD laterale.
        :param steering_limit: Limite fisico di sterzata in radianti (default 0.35 rad ≈ 20°)
        """
        self.steering_limit = steering_limit

    def compute_control(self, error_e: float, error_theta: float, kp: float, kd: float) -> float:
        """
        Calcola l'angolo di sterzo u basato sulla legge PD.
        u = Kp * e + Kd * theta_e
        """

        # 1. Calcolo dell'azione proporzionale sulla posizione
        action_p = kp * error_e

        # 2. Calcolo dell'azione "derivativa" sull'orientamento (smorzamento)
        action_d = kd * error_theta

        # 3. Somma delle componenti
        u = action_p + action_d

        # 4. Saturazione dell'output
        # Impedisce al software di richiedere angoli di sterzo fisicamente impossibili
        u_saturated = np.clip(u, -self.steering_limit, self.steering_limit)

        return u_saturated