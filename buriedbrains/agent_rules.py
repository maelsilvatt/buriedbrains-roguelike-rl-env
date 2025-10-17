# buriedbrains/agent_rules.py
from typing import Dict, Any

def create_initial_agent(name: str = "Player") -> Dict[str, Any]:
    """
    Cria e retorna o dicionário de estado para um agente iniciante.
    """
    agent = {
        'name': name,
        'level': 1,
        'hp': 300, 
        'max_hp': 300,
        'base_stats': {
            'flat_damage_bonus': 5,
            'damage_reduction': 0.0,
            'damage_modifier': 1.0 
        },
        'exp': 0,
        'exp_to_level_up': 30,
        'equipment': {},
        'skills': [],
    }
    return agent

def check_for_level_up(agent: Dict[str, Any]) -> bool:
    """
    Verifica se o agente tem experiência suficiente para subir de nível.
    """
    leveled_up = False
    while agent.get('exp', 0) >= agent.get('exp_to_level_up', float('inf')):
        leveled_up = True
        
        agent['exp'] -= agent['exp_to_level_up']
        agent['level'] += 1
        
        agent['exp_to_level_up'] = int(agent['exp_to_level_up'] * 1.05)  # Aumenta a EXP necessária para o próximo nível
                
        # --- AUMENTOS DE PODER DO AGENTE ---
        agent['max_hp'] += 30  
        agent['hp'] = agent['max_hp']
        
        # >> SUGESTÃO: Aumento de dano muito mais significativo <<
        agent['base_stats']['flat_damage_bonus'] += 5
        
        # Adiciona um pequeno bónus percentual a cada nível <<
        agent['base_stats']['damage_modifier'] += 0.02 # +2% de dano por nível

        agent['base_stats']['damage_reduction'] += 0.005 # Levemente reduzido

    return leveled_up

def instantiate_enemy(
    enemy_name: str, 
    current_floor_k: int, 
    catalogs: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Cria uma instância de um inimigo com stats escalonados pela profundidade do andar.
    """
    from . import combat

    enemies_catalog = catalogs.get('enemies', {})
    if enemy_name not in enemies_catalog:
        return None

    base_enemy_data = enemies_catalog[enemy_name].copy()
    
    # --- ESCALONAMENTO DOS INIMIGOS ---
    
    # Escalonamento de HP por andar
    hp_scaling_factor = 1 + (current_floor_k * 0.02)
    scaled_hp = int(base_enemy_data.get('hp', 50) * hp_scaling_factor)
    
    instantiated_enemy = combat.initialize_combatant(
        name=enemy_name,
        hp=scaled_hp,
        equipment=base_enemy_data.get('equipment', []),
        skills=base_enemy_data.get('skills', []),
        team=2,
        catalogs=catalogs
    )
    
    # Aumentado ligeiramente o escalonamento de dano do inimigo
    bonus_factor = (current_floor_k + 1) * 0.08
    damage_increase = int(current_floor_k * 0.8) # Um aumento mais direto: +0.8 de dano por andar.
    instantiated_enemy['base_stats']['flat_damage_bonus'] += damage_increase

    # Ajusta o yield de experiência do inimigo
    exp_increase_factor = 1 + (current_floor_k * 0.2) # +20% de EXP por andar
    instantiated_enemy['exp_yield'] = int(base_enemy_data.get('exp_yield', 20) * exp_increase_factor)

    return instantiated_enemy