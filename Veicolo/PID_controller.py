
import numpy as np


class YawRatePIDController:


    def __init__(self,
                 kp: float = 0.3,
                 ki: float = 0.0,
                 kd: float = 0.0,
                 dt: float = 0.001,
                 max_delta: float = 0.35,
                 use_integral: bool = False,
                 use_derivative: bool = False):

        self.kp = kp
        self.ki = ki if use_integral else 0.0
        self.kd = kd if use_derivative else 0.0
        self.dt = dt
        self.max_delta = max_delta

        self.use_integral = use_integral
        self.use_derivative = use_derivative

        self.integral = 0.0
        self.prev_error = 0.0


        self.integral_max = max_delta
    def compute(self, omega_des: float, omega_actual: float) -> float:


        error = omega_des - omega_actual


        p_term = self.kp * error


        i_term = 0.0
        if self.use_integral and self.ki > 0:

            self.integral += error * self.dt


            self.integral = np.clip(self.integral, -self.integral_max, self.integral_max)

            i_term = self.ki * self.integral


        d_term = 0.0
        if self.use_derivative and self.kd > 0:

            derivative = (error - self.prev_error) / self.dt
            d_term = self.kd * derivative


        delta = p_term + i_term + d_term

        delta_saturated = np.clip(delta, -self.max_delta, self.max_delta)


        if self.use_integral and abs(delta) >= self.max_delta:
            self.integral -= error * self.dt

        self.prev_error = error

        return delta_saturated

    def reset(self):

        self.integral = 0.0
        self.prev_error = 0.0

    def get_control_type(self) -> str:

        if not self.use_integral and not self.use_derivative:
            return "P"
        elif self.use_integral and not self.use_derivative:
            return "PI"
        elif not self.use_integral and self.use_derivative:
            return "PD"
        else:
            return "PID"

    def get_terms(self, omega_des: float, omega_actual: float) -> dict:

        error = omega_des - omega_actual

        p_term = self.kp * error

        i_term = 0.0
        if self.use_integral:
            i_term = self.ki * self.integral

        d_term = 0.0
        if self.use_derivative:
            derivative = (error - self.prev_error) / self.dt
            d_term = self.kd * derivative

        return {
            'error': error,
            'P': p_term,
            'I': i_term,
            'D': d_term,
            'total': p_term + i_term + d_term
        }


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