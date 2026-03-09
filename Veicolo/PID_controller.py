import numpy as np

class VelocityPIDController:


    def __init__(self, kp: float = 10.0, ki: float = 5.0, dt: float = 0.001, max_duty: float = 1.0):

        self.kp = kp
        self.ki = ki
        self.dt = dt
        self.max_duty = max_duty

        # Stato interno
        self.integral = 0.0
        self.integral_max = 1.0  # Anti-windup

    def compute(self, vx_des: float, vx_actual: float) -> float:

        # Errore
        error = vx_des - vx_actual

        # Termine P
        p_term = self.kp * error

        # Termine I
        self.integral += error * self.dt
        self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)
        i_term = self.ki * self.integral

        # Comando PI
        d_cmd = p_term + i_term

        d_cmd_saturated = np.clip(d_cmd, 0.0, self.max_duty)

        if d_cmd >= self.max_duty or d_cmd <= 0.0:
            self.integral -= error * self.dt

        return d_cmd_saturated

    def reset(self):

        self.integral = 0.0