import numpy as np
from typing import Tuple, List
from scipy.interpolate import splev, splprep


def generate_circular_path(radius: float, center: Tuple[float, float],
                           num_points: int = 150) -> List[Tuple[float, float]]:
    #Genera percorso circolare
    cx, cy = center
    angles = np.linspace(0, 2 * np.pi, num_points)

    path = []
    for angle in angles:
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        path.append((x, y))

    return path


def generate_figure_eight_path(width: float, num_points: int = 200) -> List[Tuple[float, float]]:
    #Genera percorso a forma di 8
    path = []
    t = np.linspace(0, 2 * np.pi, num_points)

    x = width * np.sin(t)
    y = width * np.sin(t) * np.cos(t)

    for i in range(num_points):
        path.append((x[i], y[i]))

    return path


def generate_circuite_path(waypoints: List[Tuple[float, float]],
                           num_points: int = 300,
                           smoothness: int = 3) -> List[Tuple[float, float]]:
    #Genera circuito interpolando waypoints con spline cubica
    waypoints_array = np.array(waypoints)
    x = waypoints_array[:, 0]
    y = waypoints_array[:, 1]

    tck, u = splprep([x, y], s=0, k=smoothness, per=1)
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    path = [(x_new[i], y_new[i]) for i in range(len(x_new))]
    return path


def get_mugello_waypoints() -> List[Tuple[float, float]]:

    #Restituisce i waypoints ispirati al Circuito del Mugello.

    return [
        (0.0, 0.0),    # Partenza / Rettilineo
        (4.0, 0.0),    # Fine rettilineo principale
        (5.5, 0.8),    # Curva 1: San Donato
        (5.0, 2.5),    # Curva 2-3: Luco - Poggio Secco
        (6.5, 3.5),    # Curva 4-5: Materassi - Borgo San Lorenzo
        (8.5, 3.0),    # Curva 6-7: Casanova - Savelli
        (10.0, 1.5),   # Curva 8-9: Arrabbiata 1 - Arrabbiata 2
        (9.0, -1.0),   # Curva 10-11: Scarperia - Palagio
        (6.0, -1.5),   # Curva 12: Correntaio
        (3.0, -2.5),   # Curva 13-14: Biondetti
        (0.5, -1.5),   # Curva 15: Bucine
        (0.0, 0.0)     # Chiusura sul traguardo
    ]

def get_trajectory(name: str) -> Tuple[List[Tuple[float, float]], str]:
    if name == 'racing':
        # chiamata ai waypoints del Mugello
        waypoints = get_mugello_waypoints()
        path = generate_circuite_path(waypoints, num_points=400)
        return path, "Circuito_Mugello"

    elif name == 'circular':
        path = generate_circular_path(radius=2.0, center=(2.0, 0.0), num_points=150)
        return path, "Traiettoria_Circolare"

    elif name == 'eight':
        path = generate_figure_eight_path(width=4.0, num_points=200)
        return path, "Traiettoria_a_8"

    else:
        raise ValueError(f"Traiettoria '{name}' non riconosciuta.")