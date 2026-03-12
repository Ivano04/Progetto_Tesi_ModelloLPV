import matplotlib.pyplot as plt
import numpy as np
import os
from Utils.trajectory_generator import get_trajectory


def save_track_images(output_dir="Immagini_Tesi"):
    # Crea la cartella se non esiste
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Elenco delle traiettorie da generare
    tracks = ['eight', 'stadium', 'racing']
    titles = ['Traiettoria a 8', 'Circuito Stadium', 'Circuito Racing']
    filenames = ['traiettoria_8.png', 'circuito_stadium.png', 'circuito_racing.png']

    for key, title, fname in zip(tracks, titles, filenames):
        # Ottieni i punti del percorso
        path, track_name = get_trajectory(key)
        path_array = np.array(path)

        plt.figure(figsize=(8, 6))
        plt.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, label='Riferimento')

        # Estetica del grafico
        plt.title(title, fontweight='bold')
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.axis('equal')  # Fondamentale per non deformare il tracciato
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Salvataggio in alta risoluzione
        save_path = os.path.join(output_dir, fname)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Salvato: {save_path}")


if __name__ == "__main__":
    save_track_images()