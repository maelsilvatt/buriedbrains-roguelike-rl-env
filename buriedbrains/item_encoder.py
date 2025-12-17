# item_encoder.py
import numpy as np

class ItemEncoder:
    """
    Traduz os atributos de um item para um vetor numérico com calibragem dinâmica.
    """
    def __init__(self, equipment_catalog: dict):
        # Tags de Efeito coesas (9 tags)
        self.effect_tags = [
            'Stun', 'Burn', 'Poison', 'Bleed', 
            'Reflect', 'Revive', 'Vulnerable', 'Blind', 'Sunder'
        ]
        
        # Tamanho do vetor: 4 atributos + 1 chance + 3 tipos + tags de efeito
        self.vector_size = 8 + len(self.effect_tags)
        
        # Executa a calibragem dinâmica
        self.max_stats = self._calibrate(equipment_catalog)

    def _calibrate(self, catalog: dict):
        """Varre o catálogo para encontrar os tetos de cada atributo."""
        max_vals = {
            'flat_damage_bonus': 1.0,   # Evita divisão por zero
            'damage_modifier': 1.0,
            'flat_hp_bonus': 1.0,
            'damage_reduction': 1.0
        }
        
        for item_data in catalog.values():
            passives = item_data.get('passive_effects', {})
            if isinstance(passives, dict):
                for stat in max_vals.keys():
                    val = float(passives.get(stat, 0))
                    if val > max_vals[stat]:
                        max_vals[stat] = val
                        
        return max_vals

    def encode(self, item_data: dict) -> np.ndarray:
        if not item_data:
            return np.zeros(self.vector_size, dtype=np.float32)
            
        vector = []
        passives = item_data.get('passive_effects', {})
        
        # Normaliza os atributos passivos
        vector.append(min(passives.get('flat_damage_bonus', 0) / self.max_stats['flat_damage_bonus'], 1.0))
        vector.append(min(passives.get('damage_modifier', 0.0) / self.max_stats['damage_modifier'], 1.0))
        vector.append(min(passives.get('flat_hp_bonus', 0) / self.max_stats['flat_hp_bonus'], 1.0))
        vector.append(min(passives.get('damage_reduction', 0.0) / self.max_stats['damage_reduction'], 1.0))
        
        # Tags de Tipo de Item
        on_hit = item_data.get('on_hit_effect', {})
        on_hurt = item_data.get('on_being_hit_effect', {})
        chance = max(on_hit.get('chance', 0), on_hurt.get('chance', 0))
        vector.append(float(chance))

        itype = item_data.get('type')
        vector.append(1.0 if itype == 'Weapon' else 0.0)
        vector.append(1.0 if itype == 'Armor' else 0.0)
        vector.append(1.0 if itype == 'Artifact' else 0.0)

        # Tags de efeito
        active_tag = on_hit.get('effect_tag') or on_hurt.get('effect_tag')
        for tag in self.effect_tags:
            vector.append(1.0 if tag == active_tag else 0.0)

        return np.array(vector, dtype=np.float32)