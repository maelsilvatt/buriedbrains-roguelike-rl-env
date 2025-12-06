# buriedbrains/effect_encoder.py
import numpy as np

class EffectEncoder:
    """
    Traduz o efeito da sala atual para um vetor One-Hot ou de Tags.
    """
    def __init__(self):
        # Lista de efeitos conhecidos (do seu YAML)
        self.known_effects = [
            'None', 
            'Slow Terrain', 'Heat', 'Dense Fog', 'Weakening Field', # Negativos
            'Evasion Zone', 'Amplifying Field', 'Consecrated Ground', 'Conductive Field' # Positivos
        ]
        self.vector_size = len(self.known_effects)

    def encode(self, effect_name: str) -> np.ndarray:
        vector = np.zeros(self.vector_size, dtype=np.float32)
        
        # Se não tiver efeito ou nome inválido, marca 'None' (índice 0)
        if not effect_name or effect_name not in self.known_effects:
            vector[0] = 1.0
            return vector
            
        # Marca o índice correspondente
        try:
            idx = self.known_effects.index(effect_name)
            vector[idx] = 1.0
        except ValueError:
            vector[0] = 1.0
            
        return vector