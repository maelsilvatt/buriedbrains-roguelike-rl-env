import numpy as np

class SkillEncoder:
    def __init__(self, skill_catalog: dict, effect_ruleset: dict):
        self.ruleset = effect_ruleset
        
        # CATEGORIAS ATUALIZADAS:
        # 'AOE' foi removido
        # 'Potency' entra como valor numérico direto no vetor, não como categoria.
        self.categories = [
            'Damage',       # Causa dano direto?
            'Heal_Self',    # Cura?
            'Buff',         # É Buff?
            'Debuff',       # É Debuff?
            'Stun',         # Atordoa?
            'DoT'           # Dano por tempo?
        ]
        
        # Tamanho do Vetor: 
        # 3 Reais (Norm_Damage, Norm_CD, Potency) + 6 Binários (Categorias)
        self.vector_size = 3 + len(self.categories)

        # Normaliza dano e CD com base no catálogo completo
        all_damages = [s.get('damage', 0) for s in skill_catalog.values()]
        all_cds = [s.get('cd', 0) for s in skill_catalog.values()]
        
        # Evita divisão por zero e pega maximos
        self.max_damage_norm = float(max([d for d in all_damages if d > 0] or [1]))
        self.max_cd_norm = float(max(all_cds or [1]))        

    def _get_effect_potency(self, tags: list) -> float:
        """
        Varre as tags da skill e busca no ruleset o maior valor float associado.
        Prioridade: 'potency' > 'stat_modifier' > 'damage_percent'.
        """
        max_pot = 0.0
        for tag in tags:
            rule = self.ruleset.get(tag)
            if not rule: continue
            
            val = 0.0
            # Valor explícito de Potência (ex: Heal, Reflect)
            if 'potency' in rule:
                val = float(rule['potency'])
                
            # Modificadores de Status (ex: Evasão 0.60, Speed -0.5)
            # Pegamos o valor absoluto (0.5 é forte, seja buff ou debuff)
            elif 'stat_modifier' in rule:
                vals = [abs(float(v)) for v in rule['stat_modifier'].values()]
                val = max(vals) if vals else 0.0
                
            # Percentuais de Dano (ex: Poison 0.04)
            elif 'damage_percent' in rule:
                val = float(rule['damage_percent'])
            
            if val > max_pot:
                max_pot = val
        return max_pot

    def encode(self, skill_data: dict) -> np.ndarray:
        """
        Shape: [Norm_Dmg, Norm_CD, Potency, Is_Dmg, Is_Heal, Is_Buff, Is_Debuff, Is_Stun, Is_DoT]
        """
        if not skill_data:
            return np.zeros(self.vector_size, dtype=np.float32)

        vector = []

        # 1. Dano Normalizado
        raw_dmg = skill_data.get('damage', 0)
        norm_dmg = max(0, raw_dmg) / self.max_damage_norm
        vector.append(min(norm_dmg, 1.0))

        # 2. CD Normalizado
        raw_cd = skill_data.get('cd', 0)
        norm_cd = raw_cd / self.max_cd_norm
        vector.append(min(norm_cd, 1.0))
                
        # Potency: Extrai o valor do ruleset (ex: 0.6 para Evasion, 0.5 para Slow)
        tags = skill_data.get('tags', [])
        potency_val = self._get_effect_potency(tags)
        vector.append(potency_val) 

        # 4. One-Hot Tags
        normalized_tags = set()
        for t in tags:
            if 'Buff' in t: normalized_tags.add('Buff')
            elif 'Debuff' in t: normalized_tags.add('Debuff')
            else: normalized_tags.add(t)

        for cat in self.categories:
            val = 1.0 if cat in normalized_tags else 0.0
            vector.append(val)

        return np.array(vector, dtype=np.float32)