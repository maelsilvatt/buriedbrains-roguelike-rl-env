import numpy as np

class ItemEncoder:
    """
    Traduz os atributos complexos de um item para um vetor numérico.
    """
    def __init__(self):
        # Tags de Efeito Relevantes para combate
        self.effect_tags = [
            'Stun', 'Burn', 'Poison', 'Bleed', 
            'Reflect', 'Revive', 'Fear', 'Shock',
            'Vulnerable', 'Blind'
        ]
        # Tamanho: [Flat_Dmg, Mod_Dmg, Flat_HP, Red_Dmg, Chance_OnHit, Is_Weapon, Is_Armor, Is_Artifact] + Tags
        self.vector_size = 8 + len(self.effect_tags)

    def encode(self, item_data: dict) -> np.ndarray:
        if not item_data:
            return np.zeros(self.vector_size, dtype=np.float32)
            
        vector = []
        
        # Stats Numéricos (Normalizados por máximos teóricos do jogo)
        passives = item_data.get('passive_effects', {})
        
        # Dano (Max Flat ~50, Max Mod ~0.5)
        vector.append(min(passives.get('flat_damage_bonus', 0) / 50.0, 1.0))
        vector.append(min(passives.get('damage_modifier', 0.0) / 0.5, 1.0))
        
        # Defesa (Max HP ~100, Max Red ~0.5)
        vector.append(min(passives.get('flat_hp_bonus', 0) / 100.0, 1.0))
        vector.append(min(passives.get('damage_reduction', 0.0) / 0.5, 1.0))
        
        # Efeitos On-Hit / On-Hurt
        on_hit = item_data.get('on_hit_effect', {})
        on_hurt = item_data.get('on_being_hit_effect', {})
        
        # Chance de proc (pega a maior entre on_hit e on_hurt)
        chance = max(on_hit.get('chance', 0), on_hurt.get('chance', 0))
        vector.append(chance)

        # Tipo do Item (One-Hot)
        itype = item_data.get('type')
        vector.append(1.0 if itype == 'Weapon' else 0.0)
        vector.append(1.0 if itype == 'Armor' else 0.0)
        vector.append(1.0 if itype == 'Artifact' else 0.0)

        # Tags de Efeito (One-Hot)
        # Junta tags de on_hit e on_hurt
        active_tag = on_hit.get('effect_tag') or on_hurt.get('effect_tag')
        
        for tag in self.effect_tags:
            vector.append(1.0 if tag == active_tag else 0.0)

        return np.array(vector, dtype=np.float32)