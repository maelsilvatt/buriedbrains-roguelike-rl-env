# buriedbrains/agent_rules.py
from typing import Dict, Any

def create_initial_agent(name: str = "Player") -> Dict[str, Any]:
    """
    Cria e retorna o dicionário de estado para um agente iniciante.
    """

    starting_skills = ["Quick Strike", "Heavy Blow", "Stone Shield", "Wait"]

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
        'karma': {
            'real': 0.0,  # Orientação (Bom/Mal)
            'imag': 0.0   # Magnitude (Força/Contexto)
        },
        'exp': 0,
        'exp_to_level_up': 30,
        'equipment': {'Artifact': 'Amulet of Vigor'
        },        
        'active_skills': starting_skills,
        'cooldowns': {skill: 0 for skill in starting_skills}
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
                        
        agent['exp_to_level_up'] += 4        
                        
        agent['max_hp'] += 30  
        agent['hp'] = agent['max_hp']
        
        agent['base_stats']['flat_damage_bonus'] += 5
        agent['base_stats']['damage_modifier'] += 0.02 # +2% de dano por nível
        agent['base_stats']['damage_reduction'] += 0.005 # Levemente reduzido

    return leveled_up

def instantiate_enemy(enemy_name: str, agent_current_floor: int, catalogs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cria uma instância de um inimigo com stats escalonados pela profundidade do andar.
    Usa as fórmulas de balanceamento da Fase 2 (Hardcore Mode).
    """
    # Importação local para evitar ciclo se 'combat' importar 'agent_rules'
    from . import combat 

    enemies_catalog = catalogs.get('enemies', {})
    enemy_base = enemies_catalog.get(enemy_name)
    
    if not enemy_base:        
        return None    
    
    # Escalonamento só começa a ter um efeito real a partir do andar 3
    effective_floor = max(0, agent_current_floor - 2)
    
    # Escalonamento de HP (8% por andar efetivo)
    hp_scaling_factor = 1 + (effective_floor * 0.08)
    scaled_hp = int(enemy_base.get('hp', 50) * hp_scaling_factor)
    
    # Inicializa o objeto de combate
    enemy_combatant = combat.initialize_combatant(
        name=enemy_name,
        hp=scaled_hp,
        equipment=enemy_base.get('equipment', []),
        skills=enemy_base.get('skills', []),
        team=2,
        catalogs=catalogs
    )
    
    # Escalonamento de Dano (12.5% por andar efetivo)
    damage_scaling_factor = 1 + (effective_floor * 0.125)
    enemy_combatant['base_stats']['flat_damage_bonus'] *= damage_scaling_factor
    
    # Escalonamento de XP (16.5% por andar efetivo)
    base_exp = enemy_base.get('exp_yield', 20)
    enemy_combatant['exp_yield'] = int(base_exp * (1 + effective_floor * 0.165))
    
    # Define nível
    enemy_combatant['level'] = agent_current_floor

    return enemy_combatant