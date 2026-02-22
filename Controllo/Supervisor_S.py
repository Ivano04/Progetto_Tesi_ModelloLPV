import numpy as np


class SupervisorS:
    def __init__(self, alpha=0.1):
        # 1. Banco dei guadagni progettati per poli reali
        # Parametri estratti dalla dinamica: m=0.041, I=27.8e-6
        self.gain_bank = {
            'LOW': {'Kp': -1.2, 'Kd': -0.2},  # Segni NEGATIVI per feedback negativo
            'MEDIUM': {'Kp': -0.8, 'Kd': -0.5},  # Aumentata magnitudo per reattività
            'HIGH': {'Kp': -0.4, 'Kd': -1.0}  # Kd alto per smorzare l'alta velocità
        }

        # 2. Definizione delle soglie con Isteresi
        # Esempio: passa a MEDIUM a 1.0 m/s, ma torna a LOW solo a 0.8 m/s
        self.thresholds = {
            'LOW_TO_MED': 1.0,
            'MED_TO_LOW': 0.8,
            'MED_TO_HIGH': 2.2,
            'HIGH_TO_MED': 2.0
        }

        self.current_mode = 'LOW'
        # --- AGGIUNTA FILTRO ---
        self.v_filtered = 0.0
        self.alpha = alpha  # Coefficiente di smoothing (0.1 = filtro forte, 0.9 = filtro leggero)
        self.type = "Standard"  # Verrà aggiornato se si usa il filtro
    def update_and_get_gains(self, vr, use_filter = False):

        #Riceve la velocità longitudinale vr e restituisce Kp, Kd
        #implementando la logica di commutazione con isteresi.
        self.type = "Filtrato" if use_filter else "Standard"

        # Calcolo della velocità da usare per la logica (reale o filtrata)
        if use_filter:
            # Formula Filtro Passa-Basso: smoothing tra il nuovo valore e il precedente
            self.v_filtered = self.alpha * abs(vr) + (1 - self.alpha) * self.v_filtered
            vr_logic = self.v_filtered
        else:
            vr_logic = abs(vr)

        if self.current_mode == 'LOW':
            if vr_logic > self.thresholds['LOW_TO_MED']:
                self.current_mode = 'MEDIUM'

        elif self.current_mode == 'MEDIUM':
            if vr_logic < self.thresholds['MED_TO_LOW']:
                self.current_mode = 'LOW'
            elif vr_logic > self.thresholds['MED_TO_HIGH']:
                self.current_mode = 'HIGH'

        elif self.current_mode == 'HIGH':
            if vr_logic < self.thresholds['HIGH_TO_MED']:
                self.current_mode = 'MEDIUM'

        # Restituisce i guadagni correnti
        gains = self.gain_bank[self.current_mode]
        return gains['Kp'], gains['Kd'], self.current_mode