# buriedbrains/skill_encoder.py
import numpy as np

class SkillEncoder:
    """
    Transforma uma Skill (dicionário do YAML) em um vetor numérico fixo
    para a Rede Neural entender 'o que' a skill faz.
    """
    def __init__(self):
        # Definição das Categorias (Tags) que importam para a tática        
        self.categories = [
            'Damage',       # Causa dano direto?
            'Heal_Self',    # Cura?
            'Buff',         # É Buff? (Defesa/Dano/Evasion)
            'Debuff',       # É Debuff? (Slow/Blind/Vulnerable)
            'Stun',         # Atordoa?
            'AOE',          # Em área?
            'DoT'           # Dano por tempo?
        ]
        self.vector_size = 2 + len(self.categories) # Dano + CD + Tags

    def encode(self, skill_data: dict) -> np.ndarray:
        """
        Retorna um vetor numpy (dtype=float32) normalizado.
        Shape: (9,)
        [Norm_Damage, Norm_CD, Is_Dmg, Is_Heal, Is_Buff, Is_Debuff, Is_Stun, Is_AOE, Is_DoT]
        """
        if not skill_data:
            return np.zeros(self.vector_size, dtype=np.float32)

        vector = []

        # 1. Dano Normalizado (Assumindo máx ~50 no jogo base)
        # Se for cura (negativo), tratamos como 0 aqui e flaggamos HEAL
        raw_dmg = skill_data.get('damage', 0)
        norm_dmg = max(0, raw_dmg) / 50.0 
        vector.append(min(norm_dmg, 1.0)) 

        # 2. Cooldown Normalizado (Assumindo máx ~10)
        raw_cd = skill_data.get('cd', 0)
        norm_cd = raw_cd / 10.0
        vector.append(min(norm_cd, 1.0))

        # 3. One-Hot Encoding das Tags
        tags = skill_data.get('tags', [])
        
        # Normaliza tags compostas (ex: Buff_Defense -> Buff)
        normalized_tags = set()
        for t in tags:
            if 'Buff' in t: normalized_tags.add('Buff')
            elif 'Debuff' in t: normalized_tags.add('Debuff')
            else: normalized_tags.add(t)

        for cat in self.categories:
            val = 1.0 if cat in normalized_tags else 0.0
            vector.append(val)

        return np.array(vector, dtype=np.float32)