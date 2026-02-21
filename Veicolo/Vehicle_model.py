import matplotlib

matplotlib.use('TkAgg')  # per finestre grafiche
import numpy as np
from dataclasses import dataclass
import math


def normalize_angle(angle: float) -> float:
    """Normalizza l'angolo tra -pi e pi."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


@dataclass
class VehicleState:
    X: float  # Posizione X globale [m]
    Y: float  # Posizione Y globale [m]
    phi: float  # Orientamento [rad]
    vx: float  # Velocità longitudinale body frame [m/s]
    vy: float  # Velocità laterale body frame [m/s]
    omega: float  # Velocità angolare [rad/s]

    def to_array(self) -> np.ndarray:
        return np.array([self.X, self.Y, self.phi, self.vx, self.vy, self.omega])

    @classmethod
    def from_array(cls, arr: np.ndarray):
        return cls(X=arr[0], Y=arr[1], phi=arr[2],
                   vx=arr[3], vy=arr[4], omega=arr[5])


@dataclass
class VehicleInput:
    d: float  # Duty cycle motore [0, 1]
    delta: float  # Angolo di sterzo [rad]

    def saturate(self):
        """Satura gli input per rispettare i limiti fisici del veicolo."""
        self.d = np.clip(self.d, 0.0, 1.0)
        self.delta = np.clip(self.delta, -0.35, 0.35)
        return self


class DynamicBicycleModel:
    def __init__(self,
                 wheelbase: float = 0.062,
                 mass: float = 0.041,
                 inertia: float = 27.8e-6,
                 lf: float = 0.029,
                 lr: float = 0.033):
        self.wb = wheelbase
        self.mass = mass
        self.inertia = inertia
        self.lf = lf
        self.lr = lr

        # Parametri motore e resistenze
        self.Cm1 = 0.287
        self.Cm2 = 0.0545
        self.Cr0 = 0.0518
        self.Cr2 = 0.00035

        # Coefficienti di rigidezza laterale
        self.Cf = 2
        self.Cr = 2

    def compute_derivatives(self, state: VehicleState, u: VehicleInput) -> np.ndarray:
        """
        Funzione di transizione di stato: calcola il vettore delle derivate (x_dot).
        Implementa il modello dinamico di bicicletta.
        """
        delta = np.clip(u.delta, -0.35, 0.35)
        d = np.clip(u.d, 0.0, 1.0)

        vx = state.vx
        vy = state.vy
        omega = state.omega
        phi = state.phi

        # --- Protezione Anti-Divergenza ---
        # Limita vx per evitare che errori numerici portino a overflow nei calcoli successivi
        if abs(vx) > 15.0:
            vx = 15.0 * np.sign(vx)

        eps = 1e-4
        vx_safe = vx if abs(vx) > eps else eps * np.sign(vx)

        # Calcolo degli angoli di deriva degli pneumatici
        alpha_f = delta - math.atan((vy + self.lf * omega) / vx_safe)
        alpha_r = - math.atan((vy - self.lr * omega) / vx_safe)

        # Forze Laterali (Modello Lineare)
        Fy_f = self.Cf * alpha_f
        Fy_r = self.Cr * alpha_r

        # Forza Longitudinale (Motore - Resistenze)
        Fx = d * self.Cm1 - d * self.Cm2 * vx - self.Cr0 * np.sign(vx) - self.Cr2 * vx * abs(vx)

        # Equazioni dinamiche nel frame body
        vx_dot = (Fx - Fy_f * math.sin(delta)) / self.mass + vy * omega
        vy_dot = (Fy_r + Fy_f * math.cos(delta)) / self.mass - vx * omega
        omega_dot = (Fy_f * self.lf * math.cos(delta) - Fy_r * self.lr) / self.inertia

        # Trasformazione cinematica nel frame globale
        X_dot = vx * math.cos(phi) - vy * math.sin(phi)
        Y_dot = vx * math.sin(phi) + vy * math.cos(phi)
        phi_dot = omega

        return np.array([X_dot, Y_dot, phi_dot, vx_dot, vy_dot, omega_dot])


class VehicleIntegrator:
    def __init__(self, model: DynamicBicycleModel, dt: float = 0.001):
        self.model = model
        self.dt = dt
        # La proprietà type viene aggiornata automaticamente in base al metodo chiamato
        self.type = "RK4"

    def Eulero(self, state: VehicleState, input: VehicleInput) -> VehicleState:
        """Integrazione di Eulero (Primo ordine)."""
        self.type = "Eulero"
        state_dot = self.model.compute_derivatives(state, input)

        x_k = state.to_array()
        x_k_plus1 = x_k + self.dt * state_dot
        x_k_plus1[2] = normalize_angle(x_k_plus1[2])

        return VehicleState.from_array(x_k_plus1)

    def RK4(self, state: VehicleState, input: VehicleInput) -> VehicleState:
        """Integrazione Runge-Kutta 4 (Quarto ordine) per maggiore precisione e stabilità."""
        self.type = "RK4"
        x = state.to_array()

        k1 = self.model.compute_derivatives(VehicleState.from_array(x), input)
        k2 = self.model.compute_derivatives(VehicleState.from_array(x + 0.5 * self.dt * k1), input)
        k3 = self.model.compute_derivatives(VehicleState.from_array(x + 0.5 * self.dt * k2), input)
        k4 = self.model.compute_derivatives(VehicleState.from_array(x + self.dt * k3), input)

        x_next = x + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next[2] = normalize_angle(x_next[2])

        return VehicleState.from_array(x_next)


class Generator_Noise:
    def __init__(self,
                 disturb_vx: bool = False,
                 disturb_omega: bool = False,
                 disturb_position: bool = False,
                 disturb_heading: bool = False,
                 magnitude: float = 0.1,
                 magnitude_position: float = 0.02,
                 magnitude_heading: float = 0.02,
                 frequency: float = 0.5,
                 disturbance_type: str = "noise"):
        self.disturb_vx = disturb_vx
        self.disturb_omega = disturb_omega
        self.disturb_position = disturb_position
        self.disturb_heading = disturb_heading
        self.magnitude = magnitude
        self.magnitude_position = magnitude_position
        self.magnitude_heading = magnitude_heading
        self.frequency = frequency
        self.disturbance_type = disturbance_type

    def get_disturbance(self, t: float, var_name: str) -> float:
        if var_name == 'vx' and not self.disturb_vx: return 0.0
        if var_name == 'omega' and not self.disturb_omega: return 0.0
        if var_name == 'position' and not self.disturb_position: return 0.0
        if var_name == 'heading' and not self.disturb_heading: return 0.0

        mag = self.magnitude_position if var_name == 'position' else \
            self.magnitude_heading if var_name == 'heading' else self.magnitude

        if self.disturbance_type == "noise":
            return np.random.normal(0, mag)
        elif self.disturbance_type == "sinusoidal":
            return mag * np.sin(2 * np.pi * self.frequency * t)
        elif self.disturbance_type == "step":
            return mag if t > 10.0 else 0.0
        elif self.disturbance_type == "impulse":
            return mag if (t % 10.0) < 0.5 else 0.0
        return 0.0